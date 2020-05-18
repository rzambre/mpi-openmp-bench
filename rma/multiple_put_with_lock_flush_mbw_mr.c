#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>

#include <mpi.h>
#include <omp.h>

#include "compute.h"

#define CACHE_LINE_SIZE		64
#define PAGE_SIZE		4096
#define DEF_NUM_THREADS		1
#define DEF_MESSAGE_SIZE        8
#define DEF_WINDOW_SHARING      1
#define DEF_COMP_THRESHOLD      1
#define WINDOW_SIZE 	        64
#define DEF_NUM_MESSAGES	640000
#define LARGE_MSG_TH	        16384
#define DEF_LARGE_NUM_MESSAGES	64000

/* An MPI+threads MPI_Put message-rate and bandwidth benchmark
 * using MPI_Win_lock/unlock with MPI_Win_flush.
 *
 * MPI+threads always with 2 processes
 * Thread i on rank 0 sends to thread i on rank 1.
 */

int num_threads;
int num_messages;
int message_size;
int way_window_sharing;
int comp_threshold;

int run_bench(int rank, int size);
void print_usage(const char *argv0);

int run_bench(int rank, int size)
{
    int i;
    int num_windows;
    size_t window_size, contig_buffer_size;
    double *t_elapsed;
    double msg_rate, my_msg_rate, bandwidth, my_bandwidth;
    MPI_Win *window;
    MPI_Info win_info;
    char *contig_buf;

    num_messages = WINDOW_SIZE * (num_messages / num_threads / WINDOW_SIZE);
    num_windows = num_threads / way_window_sharing;
     
    t_elapsed = calloc(num_threads, sizeof(double));

    /* Allocate windows */
    window = (MPI_Win *) malloc(sizeof(MPI_Win) * num_windows);

    /* Allocate contiguous buffer for all the threads on the target */
    contig_buffer_size = message_size * sizeof(char) * num_threads;
    if (rank % 2 == 0) {
        /* Putter */
        contig_buf = NULL;
    } else {
        /* Target */
        posix_memalign((void **) &contig_buf, PAGE_SIZE, contig_buffer_size);
    }

    /* Create windows depending upon the sharing */
    MPI_Info_create(&win_info);
    MPI_Info_set(win_info, "mpi_assert_new_vci", "true");
    window_size = contig_buffer_size / num_windows;
    for (i = 0; i < num_windows; i++) {
        if (rank % 2 == 0) {
            /* Putter */
            MPI_Win_create(NULL, 0, sizeof(char), win_info, MPI_COMM_WORLD, &window[i]);
        } else {
            /* Target */
            MPI_Win_create(contig_buf + i*window_size, window_size, sizeof(char), win_info, MPI_COMM_WORLD, &window[i]);
        }
    }

    for (i = 0; i < num_threads; i++)
        MPI_Win_lock_all(MPI_MODE_NOCHECK, window[i]);
    
#pragma omp parallel
    {
        int tid;
        int win_i, win_post_i, win_posts;
        int my_message_size;
        MPI_Win my_window;
        MPI_Aint target_disp;

        tid = omp_get_thread_num();
        my_message_size = message_size;
        my_window = window[tid / (num_threads / num_windows)];
        target_disp = (tid % (num_threads / num_windows)) * message_size;
        win_posts = num_messages / WINDOW_SIZE;
        if (win_posts * WINDOW_SIZE != num_messages)
            printf
                ("Warning: The final reported numbers will be off. Please choose number of messages to be a multiple of window size\n");

        if (rank % 2 == 0) {
            /* Putter */
            void *host_buf;
            double t_start, t_end;

            posix_memalign(&host_buf, PAGE_SIZE, my_message_size * sizeof(char));

            /* Warmup */
            for (win_post_i = 0; win_post_i < win_posts; win_post_i++) {
                for (win_i = 0; win_i < WINDOW_SIZE; win_i++) {
                    MPI_Put(host_buf, my_message_size, MPI_CHAR, rank + 1, target_disp, my_message_size, MPI_CHAR,
                            my_window);
                }
                MPI_Win_flush(rank + 1, my_window);
            }

#pragma omp master
            {
                MPI_Barrier(MPI_COMM_WORLD);
            }
#pragma omp barrier
            
            /* Benchmark */
            t_start = MPI_Wtime();

            for (win_post_i = 0; win_post_i < win_posts; win_post_i++) {
                for (win_i = 0; win_i < WINDOW_SIZE; win_i++) {
                    MPI_Put(host_buf, my_message_size, MPI_CHAR, rank + 1, target_disp, my_message_size, MPI_CHAR,
                              my_window);
                }
                MPI_Win_flush(rank + 1, my_window);
            }

            t_end = MPI_Wtime();
            t_elapsed[tid] = t_end - t_start;

            free(host_buf);
        } else {
            /* Target */
            
            /* Warmup */

#pragma omp master
            {
                MPI_Barrier(MPI_COMM_WORLD);
            }
#pragma omp barrier
            
            /* Benchmark */
            func(comp_threshold);
        }
    }

    for (i = 0; i < num_threads; i++)
        MPI_Win_unlock_all(window[i]);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
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
    
    MPI_Info_free(&win_info);
    for (i = 0; i < num_threads; i++)
        MPI_Win_free(&window[i]);
    free(window);
    free(contig_buf);

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
        {.name = "window-sharing",.has_arg = 1,.val = 'w'},
        {.name = "target-computation-threshold",.has_arg = 1,.val = 'C'},
        {0, 0, 0, 0}
    };

    num_threads = DEF_NUM_THREADS;
    num_messages = DEF_NUM_MESSAGES;
    message_size = DEF_MESSAGE_SIZE;
    way_window_sharing = DEF_WINDOW_SHARING;
    comp_threshold = DEF_COMP_THRESHOLD;

    while (1) {
        op = getopt_long(argc, argv, "h?T:W:M:S:w:C:", long_options, NULL);
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
            case 'w':
                way_window_sharing = atoi(optarg);
                break;
            case 'C':
                comp_threshold = atoi(optarg);
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

    if (message_size > LARGE_MSG_TH) {
        if (num_messages == DEF_NUM_MESSAGES) 
            num_messages = DEF_LARGE_NUM_MESSAGES;
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

    if (num_threads % way_window_sharing) {
        printf("The number of windows needs to be a factor of the number of threads\n");
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
    printf("  -w, --window-sharing=<way_window_sharing>	x-way sharing of RMA windows between threads\n");
    printf("  -C, --comp-threshold=<comp_threshold>	knob to control delay after which Win_free is called\n");
}
