#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>

#include <mpi.h>
#include <omp.h>

#define CACHE_LINE_SIZE		64
#define PAGE_SIZE		4096
#define DEF_MESSAGE_SIZE        8
#define WINDOW_SIZE 	        64
#define DEF_NUM_MESSAGES	640000
#define LARGE_MSG_TH	        16384
#define DEF_LARGE_NUM_MESSAGES	64000

/* MPI everywhere (process-per-core) message-rate
 * benchmark.
 *
 * Ranks of MPI everywhere with 4 processes
 * Sender       Receiver
 * 0            2
 * 1            3
 *
 */

int num_messages;
int message_size;
MPI_Comm send_recv_group;

int run_bench(int rank, int size);
void print_usage(const char *argv0);

int run_bench(int rank, int size)
{
    double t_elapsed;
    int my_pair, num_pairs, target;
    
    MPI_Request request;
    MPI_Status status;

    num_pairs = size / 2;
    my_pair = (rank < num_pairs) ? (rank) : (rank - num_pairs);

    if (rank < num_pairs) {
        /* Sender */
        void *send_buf;
        double t_start, t_end;

        target = rank + num_pairs;
        
        posix_memalign(&send_buf, PAGE_SIZE, message_size * sizeof(char));

        MPI_Issend(send_buf, message_size, MPI_CHAR, target, my_pair, MPI_COMM_WORLD,
                  &request);
        MPI_Wait(&request, &status);

        free(send_buf);
    } else {
        /* Receiver */
        void *recv_buf;
        
        target = rank - num_pairs;

        posix_memalign(&recv_buf, PAGE_SIZE, message_size * sizeof(char));

        MPI_Recv(recv_buf, message_size, MPI_CHAR, target, my_pair, MPI_COMM_WORLD,
                  &status);

        free(recv_buf);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    
    return 0;
}

int main(int argc, char *argv[])
{
    int op, ret;
    int size, rank;
    int sender_or_receiver;

    struct option long_options[] = {
        {.name = "window-size",.has_arg = 1,.val = 'W'},
        {.name = "num-messages",.has_arg = 1,.val = 'M'},
        {.name = "message-size",.has_arg = 1,.val = 'S'},
        {0, 0, 0, 0}
    };

    num_messages = DEF_NUM_MESSAGES;
    message_size = DEF_MESSAGE_SIZE;

    while (1) {
        op = getopt_long(argc, argv, "h?W:M:S:", long_options, NULL);
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
            default:
                printf("Unrecognized argument\n");
                return EXIT_FAILURE;
        }
    }

    if (optind < argc) {
        print_usage(argv[0]);
        return -1;
    }
    
    if (message_size >= LARGE_MSG_TH) {
        if (num_messages == DEF_NUM_MESSAGES) 
            num_messages = DEF_LARGE_NUM_MESSAGES;
    }

    /* MPI everywhere */
    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size % 2) {
        printf("Run with an even number of processes for MPI everywhere.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Create sender/receiver-only communicators */
    sender_or_receiver = (rank < (size / 2));
    MPI_Comm_split(MPI_COMM_WORLD, sender_or_receiver, rank, &send_recv_group);
    
    /*char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    FILE *fp;
    char path[1000];
    MPI_Get_processor_name(processor_name, &name_len);
    fp = popen("grep Cpus_allowed_list /proc/$$/status", "r");
    while (fgets(path, 1000, fp) != NULL) {
        printf("%s[%d]: %s", processor_name, rank, path);
    }*/

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
        ("  mpiexec -n 2n -ppn n -bind-to core:<#threads>1 -hosts <sender>,<receiver> %s <options>\n",
         argv0);
    printf("\n");
    printf("Options:\n");
    printf("  -M, --num-messages=<num_messages>	number of messages\n");
    printf("  -S, --message-size=<message_size>	size of messages\n");
}
