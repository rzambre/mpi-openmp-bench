#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>

#include <mpi.h>
#include <omp.h>

#define CACHE_LINE_SIZE		64
#define PAGE_SIZE		4096
#define DEF_NUM_SENDER_THREADS	1
#define DEF_MESSAGE_SIZE        8
#define WINDOW_SIZE 	        64
#define DEF_NUM_MESSAGES	640000
#define LARGE_MSG_TH	        16384
#define DEF_LARGE_NUM_MESSAGES	64000

#define MEASURE_INJECTION       1
#define FINE_ISEND              1

/* An MPI+threads (process-per-node) microbenchmark
 * with multiple sender threads that send to a single
 * receiver thread.
 *
 * The last thread is the receiver thread.
 *
 */

int num_sender_threads;
int num_messages;
int message_size;
double *t_start;
double *t_end;
double *t_elapsed;
#if FINE_ISEND
double *t_isend;
#endif
MPI_Comm *world_comms;

void receiver_thread(int rank, int target_rank);
void sender_thread(int thread_id, int rank, int target_rank, MPI_Comm my_comm);
int run_bench(int rank, int size);
void print_usage(const char *argv0);

void receiver_thread(int rank, int target_rank)
{
    void *recv_buf;
    int sender;
    int win_i, win_posts, win_post_i, last_win_post_i;
    int my_message_size;
    MPI_Request requests[WINDOW_SIZE];
    MPI_Status statuses[WINDOW_SIZE];

    my_message_size = message_size;

    posix_memalign(&recv_buf, PAGE_SIZE, my_message_size * sizeof(char));

    win_posts = num_messages / WINDOW_SIZE;
    if (win_posts * WINDOW_SIZE != num_messages)
        printf
            ("Warning: The final reported numbers will be off. Please choose number of messages to be a multiple of window size\n");

    /* Warmup */
    //printf("Receiver thread on rank %d starting warmup!\n", rank);
    for (win_post_i = 0; win_post_i < win_posts; win_post_i++) {
        for (sender = 0; sender < num_sender_threads; sender++) {
            for (win_i = 0; win_i < WINDOW_SIZE; win_i++) {
                MPI_Irecv(recv_buf, my_message_size, MPI_CHAR, target_rank, 0, world_comms[sender],
                          &requests[win_i]);
            }
            MPI_Waitall(WINDOW_SIZE, requests, statuses);
        }
    }
    //printf("Receiver thread on rank %d done with warmup!\n", rank);

    MPI_Barrier(MPI_COMM_WORLD);
    last_win_post_i = win_posts - 1;
#pragma omp barrier

    /* Benchmark */
    //printf("Receiver thread on rank %d starting benchmark!\n", rank);
    for (win_post_i = 0; win_post_i < win_posts; win_post_i++) {
        for (sender = 0; sender < num_sender_threads; sender++) {
            for (win_i = 0; win_i < WINDOW_SIZE; win_i++) {
                MPI_Irecv(recv_buf, my_message_size, MPI_CHAR, target_rank, 0, world_comms[sender],
                          &requests[win_i]);
            }
            MPI_Waitall(WINDOW_SIZE, requests, statuses);
#if !(MEASURE_INJECTION)
            if (win_post_i == last_win_post_i)
                t_end[sender] = MPI_Wtime();
#endif
        }
    }
    //printf("Receiver thread on rank %d done with benchmark!\n", rank);

    free(recv_buf);
}

void sender_thread(int thread_id, int rank, int target_rank, MPI_Comm my_comm)
{
    void *send_buf;
#if FINE_ISEND
    double t_isend_start;
#endif
    int my_message_size;
    int win_i, win_posts, win_post_i;
    MPI_Request requests[WINDOW_SIZE];
    MPI_Status statuses[WINDOW_SIZE];

    my_message_size = message_size;

    posix_memalign(&send_buf, PAGE_SIZE, my_message_size * sizeof(char));
    
    win_posts = num_messages / WINDOW_SIZE;
    if (win_posts * WINDOW_SIZE != num_messages)
        printf
            ("Warning: The final reported numbers will be off. Please choose number of messages to be a multiple of window size\n");


    /* Warmup */
    //printf("Sender %d on rank %d starting warmup!\n", thread_id, rank);
    for (win_post_i = 0; win_post_i < win_posts; win_post_i++) {
        for (win_i = 0; win_i < WINDOW_SIZE; win_i++) {
            MPI_Isend(send_buf, my_message_size, MPI_CHAR, target_rank, 0, my_comm,
                      &requests[win_i]);
        }
        MPI_Waitall(WINDOW_SIZE, requests, statuses);
    }
    //printf("Sender %d on rank %d done with warmup!\n", thread_id, rank);

#pragma omp barrier
    t_start[thread_id] = MPI_Wtime();

    /* Benchmark */
    //printf("Sender %d on rank %d starting benchmark!\n", thread_id, rank);
    for (win_post_i = 0; win_post_i < win_posts; win_post_i++) {
        for (win_i = 0; win_i < WINDOW_SIZE; win_i++) {
#if FINE_ISEND
            t_isend_start = MPI_Wtime();
#endif
            MPI_Isend(send_buf, my_message_size, MPI_CHAR, target_rank, 0, my_comm,
                      &requests[win_i]);
#if FINE_ISEND
            t_isend[thread_id] += MPI_Wtime() - t_isend_start;
#endif
        }
        MPI_Waitall(WINDOW_SIZE, requests, statuses);
    }
#if (MEASURE_INJECTION)
    t_end[thread_id] = MPI_Wtime(); 
#endif
    //printf("Sender %d on rank %d done with benchmark!\n", thread_id, rank);
    
    free(send_buf);
}

int run_bench(int rank, int size)
{
    int i;
    double msg_rate, my_msg_rate, bandwidth, my_bandwidth;

    //num_messages = WINDOW_SIZE * (num_messages / num_sender_threads / WINDOW_SIZE);

    t_start = calloc(num_sender_threads, sizeof(double));
    t_end = calloc(num_sender_threads, sizeof(double));
    t_elapsed = calloc(num_sender_threads, sizeof(double));
#if FINE_ISEND
    t_isend = calloc(num_sender_threads, sizeof(double));
#endif

    /* Create a communicator for each sender thread */
    world_comms = (MPI_Comm *) malloc(sizeof(MPI_Comm) * num_sender_threads);
    for (i = 0; i < num_sender_threads; i++) {
        MPI_Comm_dup(MPI_COMM_WORLD, &world_comms[i]);
    }

#pragma omp parallel
    {
        int tid;

        tid = omp_get_thread_num();

        if (rank % 2 == 0) {
            /* Process 0 */
            if (tid == num_sender_threads) {
                /* Receiver thread */
                receiver_thread(rank, rank + 1);
            } else {
                /* Sender thread */
                sender_thread(tid, rank, rank + 1, world_comms[tid]);
            }
        } else {
            /* Process 1 */
            if (tid == num_sender_threads) {
                /* Receiver thread */
                receiver_thread(rank, rank - 1);
            } else {
                /* Sender thread */
                sender_thread(tid, rank, rank - 1, world_comms[tid]);
            }
        }
    }

#if !(MEASURE_INJECTION)
    if (rank % 2 == 0) {
        /* Process 0: receive t_end */
        int i;
        MPI_Recv(t_end, num_sender_threads, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (i = 0; i < num_sender_threads; i++) {
            t_elapsed[i] = t_end[i] - t_start[i];
        }
    } else {
        /* Process 1: send t_end */
        MPI_Send(t_end, num_sender_threads, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
#endif

    if (rank == 0) {
        int thread_i;
#if FINE_ISEND
        double t_per_isend;
#endif

        msg_rate = 0;
        bandwidth = 0;
#if FINE_ISEND
        printf("%-10s\t%-10s\t%-10s\t%-10s\n", "Sender_thread", "Mmsgs/s", "MB/s", "Per-isend (us)");
#else
        printf("%-10s\t%-10s\t%-10s\n", "Sender_thread", "Mmsgs/s", "MB/s");
#endif
        for (thread_i = 0; thread_i < num_sender_threads; thread_i++) {
#if (MEASURE_INJECTION)
            t_elapsed[thread_i] = t_end[thread_i] - t_start[thread_i];
#endif
            my_msg_rate = ((double) num_messages / t_elapsed[thread_i]) / 1e6;
            my_bandwidth =
                (((double) message_size * (double) num_messages) / (1024 * 1024)) /
                t_elapsed[thread_i];
#if FINE_ISEND
            t_per_isend = (t_isend[thread_i] / num_messages)*1e6;
            printf("%-10d\t%-10.2f\t%-10.2f\t%f\n", thread_i, my_msg_rate, my_bandwidth, t_per_isend);
#else
            printf("%-10d\t%-10.2f\t%-10.2f\n", thread_i, my_msg_rate, my_bandwidth);
#endif
            msg_rate += my_msg_rate;
            bandwidth += my_bandwidth;
        }
        printf("\n%-10s\t%-10s\t%-10s\t%-10s\n", "Size", "Threads", "Mmsgs/s", "MB/s");
        printf("%-10d\t", message_size);
        printf("%-10d\t", num_sender_threads);
        printf("%f\t", msg_rate);
        printf("%f\n", bandwidth);
    }
    
    for (i = 0; i < num_sender_threads; i++)
        MPI_Comm_free(&world_comms[i]);
    free(world_comms);

    return 0;
}

int main(int argc, char *argv[])
{
    int op, ret;
    int provided, size, rank;

    struct option long_options[] = {
        {.name = "sender-threads",.has_arg = 1,.val = 'T'},
        {.name = "window-size",.has_arg = 1,.val = 'W'},
        {.name = "num-messages",.has_arg = 1,.val = 'M'},
        {.name = "message-size",.has_arg = 1,.val = 'S'},
        {0, 0, 0, 0}
    };

    num_sender_threads = DEF_NUM_SENDER_THREADS;
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
                num_sender_threads = atoi(optarg);
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

    /* Sender threads + 1 receiver thread */
    /* The last thread is the receiver thread */
    omp_set_num_threads(num_sender_threads + 1);

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
    printf("  -T, --sender-threads=<#threads> number of sender threads\n");
    printf("  -M, --num-messages=<num_messages>	number of messages\n");
    printf("  -S, --message-size=<message_size>	size of messages\n");
}
