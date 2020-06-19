# GPU_CUDA
These programs are using CUDA to do the General-Purpose Computation on GPU.  
1.dotproduct.cu is to compute in parallel the dot product of N = 1024*1024 random single precision floating point vectors and compare with
CPU results.  
2.dotproduct2.cu is using three kernels.  
First one is using shared memory and parallel reduction.  
Second one is shared memory, parallel reduction, and atomic function.  
Third one is shared memory, parallel reduction, and atomic lock.  
