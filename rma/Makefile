CC = icc
CFLAGS = -Wall -O3 -g3 -lmpi
MACROS =# -DERRCHK
OMPFLAGS = -fopenmp
#OMPFLAGS = -qopenmp
#DEPS = 

TARGETS=single_get_with_fence_mbw_mr
TARGETS+=single_put_with_fence_mbw_mr
TARGETS+=single_get_with_lock_flush_mbw_mr
TARGETS+=single_put_with_lock_flush_mbw_mr
TARGETS+=multiple_get_with_fence_mbw_mr
TARGETS+=multiple_put_with_fence_mbw_mr
TARGETS+=multiple_get_with_lock_flush_mbw_mr
TARGETS+=multiple_put_with_lock_flush_mbw_mr
TARGETS+=multiple_acc_with_hints_lock_flush_mbw_mr
TARGETS+=multiple_async_put_with_lock_flush_mbw_mr
TARGETS+=multiple_put_with_lock_flush_one_window_mbw_mr
TARGETS+=multiple_put_with_lock_flush_bi_mbw_mr

single_get_with_fence_mbw_mr: single_get_with_fence_mbw_mr.c
	$(CC) $(CFLAGS) $(MACROS) $^ -o $@

single_put_with_fence_mbw_mr: single_put_with_fence_mbw_mr.c
	$(CC) $(CFLAGS) $(MACROS) $^ -o $@

single_get_with_lock_flush_mbw_mr: single_get_with_lock_flush_mbw_mr.c
	$(CC) $(CFLAGS) $(MACROS) $^ -o $@

single_put_with_lock_flush_mbw_mr: single_put_with_lock_flush_mbw_mr.c compute.c
	$(CC) $(CFLAGS) $(MACROS) $^ -o $@

multiple_get_with_fence_mbw_mr: multiple_get_with_fence_mbw_mr.c
	$(CC) $(OMPFLAGS) $(CFLAGS) $(MACROS) $^ -o $@

multiple_put_with_fence_mbw_mr: multiple_put_with_fence_mbw_mr.c
	$(CC) $(OMPFLAGS) $(CFLAGS) $(MACROS) $^ -o $@

multiple_get_with_lock_flush_mbw_mr: multiple_get_with_lock_flush_mbw_mr.c
	$(CC) $(OMPFLAGS) $(CFLAGS) $(MACROS) $^ -o $@

multiple_put_with_lock_flush_mbw_mr: multiple_put_with_lock_flush_mbw_mr.c compute.c
	$(CC) $(OMPFLAGS) $(CFLAGS) $(MACROS) $^ -o $@

multiple_async_put_with_lock_flush_mbw_mr: multiple_async_put_with_lock_flush_mbw_mr.c compute.c
	$(CC) $(OMPFLAGS) $(CFLAGS) $(MACROS) $^ -o $@

multiple_put_with_lock_flush_one_window_mbw_mr: multiple_put_with_lock_flush_one_window_mbw_mr.c
	$(CC) $(OMPFLAGS) $(CFLAGS) $(MACROS) $^ -o $@

multiple_put_with_lock_flush_bi_mbw_mr: multiple_put_with_lock_flush_bi_mbw_mr.c compute.c
	$(CC) $(OMPFLAGS) $(CFLAGS) $(MACROS) $^ -o $@

multiple_acc_with_hints_lock_flush_mbw_mr: multiple_acc_with_hints_lock_flush_mbw_mr.c
	$(CC) $(OMPFLAGS) $(CFLAGS) $(MACROS) $^ -o $@

clean:
	rm -f $(TARGETS)
