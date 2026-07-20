# Exercices

1. 
a. what is the number of warps per block?
- 128 threads per block
- 32 threads per warp
128/32 = 4 warps per block

b. What is the number of warps in the grid?
- 4 warps per block
- 1024/128 = 8 blocks per grid
4x8 = 32 warp per grid

c. For the statement on line 04:
i. How many warps in the grid are active?
- 8 blocks
- 128 threads per block
- 4 warps per block
- 0 <= threadIdx.x <=  128
the first warp 0:31 respect the if statement (active)
the second 32:63 is divergent (active)
the third 64:95 is not active
the fourth 96:127 is divergent (active)
We have 3 active warps per block, which represent 3x8=24 active warps per grid

ii. How many warps in the grid are divergent?
- 2 divergent warps per block
- 8 blocks
2x8=16 divergent warps per grid

iii. What is the SIMD efficiency (in %) of warp 0 of block 0?
- warp 0 of block 0 : thread index 0:31 < 40
SIMD efficiency = 100%

iv. What is the SIMD efficiency (in %) of warp 1 of block 0?
- warp 1 of block 0 : thread index 32:39 < 40 & 40:63 > 40
SIMD efficiency = (8/32)x100 = 25%

v. What is the SIMD efficiency (in %) of warp 3 of block 0?
- warp 3 of block 0 : index 96:103<104 & 104:127>=104
SIMD efficiency = 75%

d. For the statement on line 07:
i. How many warps in the grid are active?
we have even thread index in every warps, all the 32 warps are active

ii. How many warps in the grid are divergent?
all the warps contain even & odd thread index, all the 32 warps are divergent

iii. What is the SIMD efficiency (in %) of warp 0 of block 0?
0:32 half even, half odd
SIMD efficiency = 50%

e. For the loop on line 09:
i. How many iterations have no divergence?
- max(i%3)=2 so min(5-(i%3))=3
iteration {0,1,2} have no divergence

ii. How may iterations have divergence?
(i%3) is in {0,1,2}. infact iterations {3, 4} diverge

2. For a vector addition, assume that the vector length is 2000, each thread
calculates one output element, and the thread block size is 512 threads. How
many threads will be in the grid? 
- we need 4 blocks of 512 threads to cover all 2000 outputs
4x512=2048 threads in grid

3. For the previous question, how many warps do you expect to have divergence
due to the boundary check on vector length?
we have 1 boundary condition. only 1 warp could be divergent. 2000 is not a multiple of 32 so the boundary is in the middle of a warp. There is 1 divergent warp.

4. Consider a hypothetical block with 8 threads executing a section of code
before reaching a barrier. The threads require the following amount of time
(in microseconds) to execute the sections: 2.0, 2.3, 3.0, 2.8, 2.4, 1.9, 2.6, and
2.9; they spend the rest of their time waiting for the barrier. What percentage
of the threads’ total execution time is spent waiting for the barrier?
- A = {2.0, 2.3, 3.0, 2.8, 2.4, 1.9, 2.6, 2.9}
- max A = 3.0
- wasted time : sum(maxA-effective time) = 4.1 microseconds
- total execution time : 8x3=24 microseconds
- percentage = 17%

5. A CUDA programmer says that if they launch a kernel with only 32 threads
in each block, they can leave out the __syncthreads() instruction wherever
barrier synchronization is needed. Do you think this is a good idea? Explain.
L'intuition repose sur le lock-step SIMD du warp (synchronisation implicite supposée). Mais la divergence, et depuis Volta l'independent thread scheduling, font qu'on ne peut pas supposer un timing identique entre threads d'un même warp. Il faut donc conserver une synchronisation explicite __syncthreads(), ou __syncwarp() pour un scope warp.

6. If a CUDA device’s SM can take up to 1536 threads and up to 4 thread
blocks, which of the following block configurations would result in the most
number of threads in the SM?
a. 128 threads per block
b. 256 threads per block
c. 512 threads per block
d. 1024 threads per block
- We observe that 512x3=1536. We conclude that 512 threads equal an occupancy of 100%.
- We could have calculated all occupancies and pick the max


7. Assume a device that allows up to 64 blocks per SM and 2048 threads per
SM. Indicate which of the following assignments per SM are possible. In the
cases in which it is possible, indicate the occupancy level.
a. 8 blocks with 128 threads each
b. 16 blocks with 64 threads each
c. 32 blocks with 32 threads each
d. 64 blocks with 32 threads each
e. 32 blocks with 64 threads each
- Options {a,b,c,d,e} are possible and have respectively an occupancy level of {50%, 50%, 50%, 100%, 100%}

8. Consider a GPU with the following hardware limits: 2048 threads per SM, 32
blocks per SM, and 64K (65,536) registers per SM. For each of the following
kernel characteristics, specify whether the kernel can achieve full occupancy.
If not, specify the limiting factor.
a. The kernel uses 128 threads per block and 30 registers per thread.
b. The kernel uses 32 threads per block and 29 registers per thread.
c. The kernel uses 256 threads per block and 34 registers per thread.
- a. with 16 blocks per SM we can achieve 100% occupancy
- b. The kernel specification needs more blocks per SM to achieve 100% occupancy (50% currently)
- c. 34x2048 = 69632 registers > max. The kernel cant achieve 100%.

9. A student mentions that they were able to multiply two 1024x1024 matrices
using a matrix multiplication kernel with 32x32 thread blocks. The student is
using a CUDA device that allows up to 512 threads per block and up to 8 blocks
per SM. The student further mentions that each thread in a thread block calculates
one element of the result matrix. What would be your reaction and why?
- He is lying. 32x32=1024 threads per block which is not supported in this case.
