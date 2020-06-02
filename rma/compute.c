#include "compute.h"

void func(int64_t num_iter)
{
#if 1
    /* SSE */
    asm volatile ("xorps       %%xmm0, %%xmm0\n\t"
                  "xorps       %%xmm1, %%xmm1\n\t"
                  "loop_head_%=:\n\t"
                  "mulps       %%xmm1, %%xmm0\n\t"
                  "decq        %0\n\t"
                  "jne         loop_head_%=\n":"+r" (num_iter)::"cc", "xmm0", "xmm1");
#else
    /* AVX */
    asm volatile ("vxorps       %%ymm0, %%ymm0, %%ymm0\n\t"
                  "vxorps       %%ymm1, %%ymm1, %%ymm1\n\t"
                  "vxorps       %%ymm2, %%ymm2, %%ymm2\n\t"
                  "loop_head_%=:\n\t"
                  "vfmadd132ps  %%ymm0, %%ymm1, %%ymm2\n\t"
                  "decq         %0\n\t"
                  "jne          loop_head_%=\n":"+r" (num_iter)::"cc", "ymm0", "ymm1", "ymm2");
#endif
}
