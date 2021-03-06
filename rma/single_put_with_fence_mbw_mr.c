#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>

#include <mpi.h>
#include <omp.h>

#define CACHE_LINE_SIZE		64
#define PAGE_SIZE		4096
#define DEF_NUM_THREADS		1
#define DEF_MESSAGE_SIZE        8
#define DEF_LOOP_THRESHOLD      0
#define WINDOW_SIZE 	        64
#define DEF_NUM_MESSAGES	1000000

/* An MPI everywere MPI_Put message-rate and bandwidth benchmark
 * using MPI_Win_fence.
 *
 * MPI everywhere with even number of processes. First n/2 ranks
 * on node 1 and the second n/2 ranks on node 2.
 */

int num_messages;
int message_size;
int num_pairs;
int loop_threshold;
MPI_Comm initiator_target_group;
MPI_Comm my_pair_group;

int run_bench(int rank, int size);
void print_usage(const char *argv0);

int run_bench(int rank, int size)
{
    double t_elapsed, *ts_elapsed = NULL;
    int win_i, win_post_i, win_posts;
    char *target_buf;
    
    MPI_Win window;

    if (rank < num_pairs) {
        /* Putter */
        target_buf = NULL;
        MPI_Win_create(NULL, 0, sizeof(char), MPI_INFO_NULL, my_pair_group, &window);
    } else {
        /* Target */
        posix_memalign((void **) &target_buf, PAGE_SIZE, message_size * sizeof(char));
        MPI_Win_create(target_buf, message_size * sizeof(char), sizeof(char), MPI_INFO_NULL, my_pair_group, &window);
    }

    win_posts = num_messages / WINDOW_SIZE;
    if (win_posts * WINDOW_SIZE != num_messages)
        printf
            ("Warning: The final reported numbers will be off. Please choose number of messages to be a multiple of window size\n");

    if (rank < num_pairs) {
        /* Putter */
        void *host_buf;
        double t_start, t_end;

        posix_memalign(&host_buf, PAGE_SIZE, message_size * sizeof(char));

        /* Warmup */
        for (win_post_i = 0; win_post_i < win_posts; win_post_i++) {
            MPI_Win_fence(0, window);
            for (win_i = 0; win_i < WINDOW_SIZE; win_i++) {
                MPI_Put(host_buf, message_size, MPI_CHAR, 1, 0, message_size, MPI_CHAR,
                        window);
            }
            MPI_Win_fence(0, window);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        
        /* Benchmark */
        t_start = MPI_Wtime();

        for (win_post_i = 0; win_post_i < win_posts; win_post_i++) {
            MPI_Win_fence(0, window);
            for (win_i = 0; win_i < WINDOW_SIZE; win_i++) {
                MPI_Put(host_buf, message_size, MPI_CHAR, 1, 0, message_size, MPI_CHAR,
                          window);
            }
            MPI_Win_fence(0, window);
        }

        t_end = MPI_Wtime();
        t_elapsed = t_end - t_start;

        free(host_buf);
    } else {
        /* Target */
        
        /* Warmup */
        for (win_post_i = 0; win_post_i < win_posts; win_post_i++) {
            MPI_Win_fence(0, window);
            MPI_Win_fence(0, window);
        
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        /* Benchmark */
        for (win_post_i = 0; win_post_i < win_posts; win_post_i++) {
            MPI_Win_fence(0, window);
            
            /* Fake computation */
            sleep(loop_threshold);
            
            MPI_Win_fence(0, window);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0)
        ts_elapsed = calloc(num_pairs, sizeof(double));
    if (rank < (size / 2))
        MPI_Gather(&t_elapsed, 1, MPI_DOUBLE, ts_elapsed, 1, MPI_DOUBLE, 0, initiator_target_group);
    if (rank == 0) {
        int pair_i;
        double pair_msg_rate, msg_rate, pair_bandwidth, bandwidth;

        msg_rate = 0;
        bandwidth = 0;
        printf("%-10s\t%-10s\t%-10s\n", "Pair", "Mmsgs/s", "MB/s");
        for (pair_i = 0; pair_i < (size / 2); pair_i++) {
            pair_msg_rate = ((double) num_messages / ts_elapsed[pair_i]) / 1e6;
            pair_bandwidth =
                (((double) message_size * (double) num_messages) / (1024 * 1024)) /
                ts_elapsed[pair_i];
            printf("%-10d\t%-10.2f\t%-10.2f\n", pair_i, pair_msg_rate, pair_bandwidth);
            msg_rate += pair_msg_rate;
            bandwidth += pair_bandwidth;
        }
        printf("\n%-10s\t%-10s\t%-10s\t%-10s\n", "Size", "Pairs", "Mmsgs/s", "MB/s");
        printf("%-10d\t", message_size);
        printf("%-10d\t", (size / 2));
        printf("%f\t", msg_rate);
        printf("%f\n", bandwidth);

        free(ts_elapsed);
    }
    
    MPI_Win_free(&window);
    if (target_buf)
        free(target_buf);

    return 0;
}

int main(int argc, char *argv[])
{
    int op, ret;
    int size, rank;
    int my_pair, initiator_or_target;

    struct option long_options[] = {
        {.name = "window-size",.has_arg = 1,.val = 'W'},
        {.name = "num-messages",.has_arg = 1,.val = 'M'},
        {.name = "message-size",.has_arg = 1,.val = 'S'},
        {.name = "target-computation-length",.has_arg = 1,.val = 'C'},
        {0, 0, 0, 0}
    };

    num_messages = DEF_NUM_MESSAGES;
    message_size = DEF_MESSAGE_SIZE;
    loop_threshold = DEF_LOOP_THRESHOLD;

    while (1) {
        op = getopt_long(argc, argv, "h?W:M:S:C:", long_options, NULL);
        if (op == -1)
            break;

        switch (op) {
            case '?':
            case 'h':
                print_usage(argv[0]);
                return -1;
            case 'M':
                num_messages = atoi(optarg);
                break;
            case 'S':
                message_size = atoi(optarg);
                break;
            case 'C':
                loop_threshold = atoi(optarg);
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
    
    /* MPI everywhere */
    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size % 2) {
        printf("Run with an even number of processes for MPI everywhere.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    num_pairs = size / 2;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Create initiator/target-only communicators */
    initiator_or_target = (rank < num_pairs);
    MPI_Comm_split(MPI_COMM_WORLD, initiator_or_target, rank, &initiator_target_group);
    
    /* Create a communicator for each pair */
    my_pair = (rank < num_pairs) ? (rank) : (rank - num_pairs);
    MPI_Comm_split(MPI_COMM_WORLD, my_pair, rank, &my_pair_group);

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
        ("  mpiexec -n 2n -ppn n -bind-to core:<#threads> -hosts <initiator>,<target> %s <options>\n",
         argv0);
    printf("\n");
    printf("Options:\n");
    printf("  -M, --num-messages=<num_messages>	number of messages\n");
    printf("  -S, --message-size=<message_size>	size of messages\n");
}
