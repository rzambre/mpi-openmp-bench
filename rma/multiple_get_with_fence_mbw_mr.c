#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>

#include <mpi.h>
#include <omp.h>

#define CACHE_LINE_SIZE		64
#define PAGE_SIZE		4096
#define DEF_NUM_THREADS		1
#define DEF_MESSAGE_SIZE        8
#define WINDOW_SIZE 	        64
#define DEF_NUM_MESSAGES	1000000

/* An MPI+threads MPI_Get message-rate and bandwidth benchmark
 * using MPI_Win_flush.
 *
 * MPI+threads always with 2 processes
 * Thread i on rank 0 sends to thread i on rank 1.
 */

int num_threads;
int num_messages;
int message_size;

int run_bench(int rank, int size);
void print_usage(const char *argv0);

int run_bench(int rank, int size)
{
    int i;
    double *t_elapsed;
    double msg_rate, my_msg_rate, bandwidth, my_bandwidth;
    MPI_Win *window;
    char **target_buf;

    num_messages = WINDOW_SIZE * (num_messages / num_threads / WINDOW_SIZE);

    t_elapsed = calloc(num_threads, sizeof(double));

    /* Create a window for each thread */
    window = (MPI_Win *) malloc(sizeof(MPI_Win) * num_threads);
    target_buf = (char **) malloc(sizeof(char *) * num_threads);
    for (i = 0; i < num_threads; i++) {
        if (rank % 2 == 0) {
            /* Getter */
            target_buf[i] = NULL;
            MPI_Win_create(NULL, 0, sizeof(char), MPI_INFO_NULL, MPI_COMM_WORLD, &window[i]);
        } else {
            /* Target */
            posix_memalign((void **) &target_buf[i], PAGE_SIZE, message_size * sizeof(char));
            MPI_Win_create(target_buf[i], message_size * sizeof(char), sizeof(char), MPI_INFO_NULL, MPI_COMM_WORLD, &window[i]);
        }
    }

#pragma omp parallel
    {
        int tid;
        int win_i, win_post_i, win_posts;
        int my_message_size;

        tid = omp_get_thread_num();
        my_message_size = message_size;

        win_posts = num_messages / WINDOW_SIZE;
        if (win_posts * WINDOW_SIZE != num_messages)
            printf
                ("Warning: The final reported numbers will be off. Please choose number of messages to be a multiple of window size\n");

        if (rank % 2 == 0) {
            /* Getter */
            void *recv_buf;
            double t_start, t_end;

            posix_memalign(&recv_buf, PAGE_SIZE, my_message_size * sizeof(char));

            /* Warmup */
            for (win_post_i = 0; win_post_i < win_posts; win_post_i++) {
                MPI_Win_fence(0, window[tid]);
                for (win_i = 0; win_i < WINDOW_SIZE; win_i++) {
                    MPI_Get(recv_buf, my_message_size, MPI_CHAR, rank + 1, 0, my_message_size, MPI_CHAR,
                            window[tid]);
                }
                MPI_Win_fence(0, window[tid]);
            }

#pragma omp master
            {
                MPI_Barrier(MPI_COMM_WORLD);
            }
#pragma omp barrier
            
            /* Benchmark */
            t_start = MPI_Wtime();

            for (win_post_i = 0; win_post_i < win_posts; win_post_i++) {
                MPI_Win_fence(0, window[tid]);
                for (win_i = 0; win_i < WINDOW_SIZE; win_i++) {
                    MPI_Get(recv_buf, my_message_size, MPI_CHAR, rank + 1, 0, my_message_size, MPI_CHAR,
                              window[tid]);
                }
                MPI_Win_fence(0, window[tid]);
            }

            t_end = MPI_Wtime();
            t_elapsed[tid] = t_end - t_start;

            free(recv_buf);
        } else {
            /* Target */
            
            /* Warmup */
            for (win_post_i = 0; win_post_i < win_posts; win_post_i++) {
                MPI_Win_fence(0, window[tid]);
                MPI_Win_fence(0, window[tid]);
            
            }
#pragma omp master
            {
                MPI_Barrier(MPI_COMM_WORLD);
            }
#pragma omp barrier
            
            /* Benchmark */
            for (win_post_i = 0; win_post_i < win_posts; win_post_i++) {
                MPI_Win_fence(0, window[tid]);
                MPI_Win_fence(0, window[tid]);
            }
        }
    }

    if (rank % 2 == 0) {
        int thread_i;
        msg_rate = 0;
        bandwidth = 0;
        printf("%-10s\t%-10s\t%-10s\n", "Thread", "Mmsgs/s", "MB/s");
        for (thread_i = 0; thread_i < num_threads; thread_i++) {
            my_msg_rate = ((double) num_messages / t_elapsed[thread_i]) / 1e6;
            my_bandwidth =
                (((double) message_size * (double) num_messages) / (1024 * 1024)) /
                t_elapsed[thread_i];
            printf("%-10d\t%-10.2f\t%-10.2f\n", thread_i, my_msg_rate, my_bandwidth);
            msg_rate += my_msg_rate;
            bandwidth += my_bandwidth;
        }
        printf("\n%-10s\t%-10s\t%-10s\t%-10s\n", "Size", "Threads", "Mmsgs/s", "MB/s");
        printf("%-10d\t", message_size);
        printf("%-10d\t", num_threads);
        printf("%f\t", msg_rate);
        printf("%f\n", bandwidth);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    
    for (i = 0; i < num_threads; i++) {
        if (target_buf[i])
            free(target_buf[i]);
        MPI_Win_free(&window[i]);
    }
    free(window);
    if (target_buf)
        free(target_buf);

    return 0;
}

int main(int argc, char *argv[])
{
    int op, ret;
    int provided, size, rank;

    struct option long_options[] = {
        {.name = "threads",.has_arg = 1,.val = 'T'},
        {.name = "window-size",.has_arg = 1,.val = 'W'},
        {.name = "num-messages",.has_arg = 1,.val = 'M'},
        {.name = "message-size",.has_arg = 1,.val = 'S'},
        {0, 0, 0, 0}
    };

    num_threads = DEF_NUM_THREADS;
    num_messages = DEF_NUM_MESSAGES;
    message_size = DEF_MESSAGE_SIZE;

    while (1) {
        op = getopt_long(argc, argv, "h?T:W:M:S:", long_options, NULL);
        if (op == -1)
            break;

        switch (op) {
            case '?':
            case 'h':
                print_usage(argv[0]);
                return -1;
            case 'T':
                num_threads = atoi(optarg);
                break;
            case 'M':
                num_messages = atoi(optarg);
                break;
            case 'S':
                message_size = atoi(optarg);
                break;
            default:
                printf("Unrecognized argument\n");
                return EXIT_FAILURE;
        }
    }

    if (optind < argc) {
        print_usage(argv[0]);
        return -1;
    }

    /* MPI+threads */
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided != MPI_THREAD_MULTIPLE) {
        printf("Thread multiple needed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 2) {
        printf("Run with only two processes for MPI+threads.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    omp_set_num_threads(num_threads);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    ret = run_bench(rank, size);
    if (ret) {
        fprintf(stderr, "Error in running bench \n");
        ret = EXIT_FAILURE;
    }

    MPI_Finalize();

    return ret;
}

void print_usage(const char *argv0)
{
    printf("Usage:\n");
    printf
        ("  mpiexec -n 2 -ppn 1 -bind-to core:<#threads> -hosts <sender>,<receiver> %s <options>\n",
         argv0);
    printf("\n");
    printf("Options:\n");
    printf("  -T, --threads=<#threads>			number of threads\n");
    printf("  -M, --num-messages=<num_messages>	number of messages\n");
    printf("  -S, --message-size=<message_size>	size of messages\n");
}
