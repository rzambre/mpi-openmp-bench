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
#define DEF_NUM_MESSAGES	640000
#define LARGE_MSG_TH	        16384
#define DEF_LARGE_NUM_MESSAGES	64000


/* An MPI+threads (process-per-node) message-rate
 * benchmark with FUNNELED mode (message size depends
 * on number of threads). Does not accommodate
 * varying levels of hybrid.
 *
 * MPI+threads always with 2 processes
 * Thread i on rank 0 sends to thread i on rank 1.
 */

int num_threads;
int num_messages;
int message_size;
MPI_Comm send_recv_group;

int run_bench(int rank, int size);
void print_usage(const char *argv0);

int run_bench(int rank, int size)
{
    double t_elapsed;
    double msg_rate, bandwidth;

    //num_messages = WINDOW_SIZE * (num_messages / num_threads / WINDOW_SIZE);

    int my_rank, my_tag;
    MPI_Comm my_comm;
    int win_i, win_post_i, win_posts;
    int my_message_size;
    int sync_buf;
    MPI_Request requests[WINDOW_SIZE];
    MPI_Status statuses[WINDOW_SIZE];

    my_rank = rank;
    my_tag = 0;
    my_comm = MPI_COMM_WORLD;
    my_message_size = message_size * num_threads;

    win_posts = num_messages / WINDOW_SIZE;
    if (win_posts * WINDOW_SIZE != num_messages)
        printf
            ("Warning: The final reported numbers will be off. Please choose number of messages to be a multiple of window size\n");

    if (my_rank % 2 == 0) {
        /* Sender */
        void *send_buf;
        double t_start, t_end;

        posix_memalign(&send_buf, PAGE_SIZE, my_message_size * sizeof(char));

        /* Warmup */
        for (win_post_i = 0; win_post_i < win_posts; win_post_i++) {
            for (win_i = 0; win_i < WINDOW_SIZE; win_i++) {
                MPI_Isend(send_buf, my_message_size, MPI_CHAR, my_rank + 1, my_tag, my_comm,
                          &requests[win_i]);
            }
            MPI_Waitall(WINDOW_SIZE, requests, statuses);
        }

        MPI_Recv(&sync_buf, 1, MPI_INT, my_rank + 1, my_tag, my_comm, MPI_STATUS_IGNORE);

        MPI_Barrier(MPI_COMM_WORLD);
        
        t_start = MPI_Wtime();

        for (win_post_i = 0; win_post_i < win_posts; win_post_i++) {
            for (win_i = 0; win_i < WINDOW_SIZE; win_i++) {
                MPI_Isend(send_buf, my_message_size, MPI_CHAR, my_rank + 1, my_tag, my_comm,
                          &requests[win_i]);
            }
            MPI_Waitall(WINDOW_SIZE, requests, statuses);
        }

        MPI_Recv(&sync_buf, 1, MPI_INT, my_rank + 1, my_tag, my_comm, MPI_STATUS_IGNORE);

        t_end = MPI_Wtime();

        t_elapsed = t_end - t_start;

        free(send_buf);
    } else {
        /* Receiver */
        void *recv_buf;

        posix_memalign(&recv_buf, PAGE_SIZE, my_message_size * sizeof(char));

        /* Warmup */
        for (win_post_i = 0; win_post_i < win_posts; win_post_i++) {
            for (win_i = 0; win_i < WINDOW_SIZE; win_i++) {
                MPI_Irecv(recv_buf, my_message_size, MPI_CHAR, my_rank - 1, my_tag, my_comm,
                          &requests[win_i]);
            }
            MPI_Waitall(WINDOW_SIZE, requests, statuses);
        }

        MPI_Send(&sync_buf, 1, MPI_INT, my_rank - 1, my_tag, my_comm);

        MPI_Barrier(MPI_COMM_WORLD);

        for (win_post_i = 0; win_post_i < win_posts; win_post_i++) {
            for (win_i = 0; win_i < WINDOW_SIZE; win_i++) {
                MPI_Irecv(recv_buf, my_message_size, MPI_CHAR, my_rank - 1, my_tag, my_comm,
                          &requests[win_i]);
            }
            MPI_Waitall(WINDOW_SIZE, requests, statuses);
        }

        MPI_Send(&sync_buf, 1, MPI_INT, my_rank - 1, my_tag, my_comm);

        free(recv_buf);
    }

    if (rank == 0) {
        msg_rate = ((double) num_messages / t_elapsed) / 1e6;
        bandwidth =
            (((double) message_size * (double) num_messages) / (1024 * 1024)) /
            t_elapsed;
        printf("\n%-10s\t%-10s\t%-10s\t%-10s\n", "Size", "Threads", "Mmsgs/s", "MB/s");
        printf("%-10d\t", message_size);
        printf("%-10d\t", num_threads);
        printf("%f\t", msg_rate);
        printf("%f\n", bandwidth);
    }
    
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
        op = getopt_long(argc, argv, "h?T:W:M:S:s:", long_options, NULL);
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
            case 's':
                /* This is a quick hack; script needs to be updated */
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
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided != MPI_THREAD_FUNNELED) {
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
