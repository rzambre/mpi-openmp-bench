CC = icc
CFLAGS = -Wall -O3 -g3 -lmpi
MACROS =# -DERRCHK
OMPFLAGS = -fopenmp
#OMPFLAGS = -qopenmp
#DEPS = 

TARGETS=single_isir_waitall_mbw_mr
TARGETS+=single_isir_testall_mbw_mr
TARGETS+=single_isiprober_testall_mbw_mr
TARGETS+=single_isir_test_ping
TARGETS+=funneled_isir_waitall_comm_world_mbw_mr
TARGETS+=serialized_isir_waitall_comm_world_mbw_mr
TARGETS+=multiple_isir_waitall_mbw_mr
TARGETS+=multiple_isir_testall_mbw_mr
TARGETS+=multiple_isiprober_testall_mbw_mr
TARGETS+=multiple_isir_testall_mr_toy
TARGETS+=multiple_isir_waitall_comm_world_mbw_mr
TARGETS+=multiple_isir_test_ping
TARGETS+=multiple_multi_sender_single_receiver
TARGETS+=multiple_single_sender_multi_receiver_mbw_mr

single_isir_waitall_mbw_mr: single_isir_waitall_mbw_mr.c
	$(CC) $(CFLAGS) $(MACROS) $^ -o $@

single_isir_testall_mbw_mr: single_isir_testall_mbw_mr.c
	$(CC) $(CFLAGS) $(MACROS) $^ -o $@

single_isiprober_testall_mbw_mr: single_isiprober_testall_mbw_mr.c
	$(CC) $(CFLAGS) $(MACROS) $^ -o $@

single_isir_test_ping: single_isir_test_ping.c
	$(CC) $(CFLAGS) $(MACROS) $^ -o $@

funneled_isir_waitall_comm_world_mbw_mr: funneled_isir_waitall_comm_world_mbw_mr.c
	$(CC) $(OMPFLAGS) $(CFLAGS) $(MACROS) $^ -o $@

serialized_isir_waitall_comm_world_mbw_mr: serialized_isir_waitall_comm_world_mbw_mr.c
	$(CC) $(OMPFLAGS) $(CFLAGS) $(MACROS) $^ -o $@

multiple_isir_waitall_mbw_mr: multiple_isir_waitall_mbw_mr.c
	$(CC) $(OMPFLAGS) $(CFLAGS) $(MACROS) $^ -o $@

multiple_isir_testall_mbw_mr: multiple_isir_testall_mbw_mr.c
	$(CC) $(OMPFLAGS) $(CFLAGS) $(MACROS) $^ -o $@

multiple_isiprober_testall_mbw_mr: multiple_isiprober_testall_mbw_mr.c
	$(CC) $(OMPFLAGS) $(CFLAGS) $(MACROS) $^ -o $@

multiple_isir_testall_mr_toy: multiple_isir_testall_mr_toy.c
	$(CC) $(OMPFLAGS) $(CFLAGS) $(MACROS) $^ -o $@

multiple_isir_waitall_comm_world_mbw_mr: multiple_isir_waitall_comm_world_mbw_mr.c
	$(CC) $(OMPFLAGS) $(CFLAGS) $(MACROS) $^ -o $@

multiple_isir_test_ping: multiple_isir_test_ping.c
	$(CC) $(OMPFLAGS) $(CFLAGS) $(MACROS) $^ -o $@

multiple_multi_sender_single_receiver: multiple_multi_sender_single_receiver.c
	$(CC) $(OMPFLAGS) $(CFLAGS) $(MACROS) $^ -o $@

multiple_single_sender_multi_receiver_mbw_mr: multiple_single_sender_multi_receiver_mbw_mr.c
	$(CC) $(OMPFLAGS) $(CFLAGS) $(MACROS) $^ -o $@

clean:
	rm -f $(TARGETS)
