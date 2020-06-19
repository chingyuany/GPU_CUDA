#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define N 1 << 24
#define threads_per_block 512

struct Lock {
  int *mutex;
  Lock(void) {
    int state = 0;
    cudaMalloc((void **)&mutex, sizeof(int));
    cudaMemcpy(mutex, &state, sizeof(int), cudaMemcpyHostToDevice);
  }

  ~Lock(void) { cudaFree(mutex); }

  __device__ void lock(void) {
    while (atomicCAS(mutex, 0, 1) != 0)
      ;
  }

  __device__ void unlock(void) { atomicExch(mutex, 0); }
};

__global__ void GPU_big_dot(float *a, float *b, float *c, int n) {
  //  set thread ID
  unsigned int tid = threadIdx.x;
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // covert global data pointer to the local a and b array 's pointer of this
  // block
  float *ia = a + blockIdx.x * blockDim.x;
  float *ib = b + blockIdx.x * blockDim.x;
  // boundary check
  if (idx >= n)
    return;
  // declare shared memory
  __shared__ float shared[threads_per_block];
  // put resultt to the shared memory
  shared[tid] = ia[tid] * ib[tid];
  __syncthreads();
  // in-place reduction in shared memory
  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    if ((tid % (2 * stride)) == 0) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  // write result for this block in shared memory to global mem
  if (tid == 0) {
    c[blockIdx.x] = shared[0];
  }
}

__global__ void atomic_function_GPU_big_dot(float *a, float *b, float *c,
                                            int n) {
  //  set thread ID
  unsigned int tid = threadIdx.x;
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // covert global data pointer to the local a and b array 's pointer of this
  // block
  float *ia = a + blockIdx.x * blockDim.x;
  float *ib = b + blockIdx.x * blockDim.x;
  // boundary check
  if (idx >= n)
    return;
  // declare shared memory
  __shared__ float shared[threads_per_block];
  // put resultt to the shared memory
  shared[tid] = ia[tid] * ib[tid];
  __syncthreads();
  // in-place reduction in shared memory
  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    if ((tid % (2 * stride)) == 0) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  // write result for this block in shared memory to global mem
  if (tid == 0) {
    atomicAdd(c, shared[0]);
  }
}

__global__ void atomic_lock_GPU_big_dot(float *a, float *b, float *c, int n,
                                        Lock lock) {
  //  set thread ID
  unsigned int tid = threadIdx.x;
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // covert global data pointer to the local a and b array 's pointer of this
  // block
  float *ia = a + blockIdx.x * blockDim.x;
  float *ib = b + blockIdx.x * blockDim.x;
  // boundary check
  if (idx >= n)
    return;
  // declare shared memory
  __shared__ float shared[threads_per_block];
  // put resultt to the shared memory
  shared[tid] = ia[tid] * ib[tid];
  __syncthreads();
  // in-place reduction in shared memory
  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    if ((tid % (2 * stride)) == 0) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  // write result for this block in shared memory to global mem

  if (tid == 0) {
    lock.lock();
    c[0] += shared[0];
    lock.unlock();
  }
}

void random_vecotr_init(float *vector) {

  for (int i = 0; i < N; i++) {
    vector[i] = (float)rand() / RAND_MAX;
  }
}

int main(int argc, char *argv[]) {
  float *vector1, *vector2, *d_v1, *d_v2, *d_result, *d_result_atomic_fun,
      *d_result_atomic_lock, *GPUResult, *GPUResult_atmoic_function,
      *GPUResult_atmoic_lock, gpuSum = 0.0;

  size_t vector_size = sizeof(float) * N;

  // allocate memory space for host memory
  vector1 = (float *)malloc(vector_size);
  vector2 = (float *)malloc(vector_size);

  // init two vectors with random float numbers
  srand(time(NULL));
  random_vecotr_init(vector1);
  random_vecotr_init(vector2);
  // print_vector(vector1, (char *)"vector1");
  // print_vector(vector2, (char *)"vector2");

  // define grid and block size
  dim3 block(threads_per_block, 1);
  dim3 grid((N + block.x - 1) / block.x, 1);
  printf("grid= %d block= %d\n", grid.x, block.x);

  // capture the start time
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // allocate memory space for device memory
  cudaMalloc((void **)&d_v1, vector_size);
  cudaMalloc((void **)&d_v2, vector_size);
  cudaMalloc((void **)&d_result, grid.x * sizeof(float));
  cudaMalloc((void **)&d_result_atomic_fun, sizeof(float));
  cudaMalloc((void **)&d_result_atomic_lock, sizeof(float));
  GPUResult = (float *)malloc(grid.x * sizeof(float));
  GPUResult_atmoic_function = (float *)malloc(sizeof(float));
  GPUResult_atmoic_lock = (float *)malloc(sizeof(float));

  // Kernel1 :shared memory and parallel reduction
  // copy vectors from host to device
  cudaMemcpy(d_v1, vector1, vector_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_v2, vector2, vector_size, cudaMemcpyHostToDevice);

  // launch kernel
  cudaEventRecord(start, 0);
  GPU_big_dot<<<grid, block>>>(d_v1, d_v2, d_result, N);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsedTime1, elapsedTime2, elapsedTime3;
  cudaEventElapsedTime(&elapsedTime1, start, stop);
  printf("Kernel no atomic execution time:  %3.10f sec\n", elapsedTime1 / 1000);

  // copy result back to host
  cudaMemcpy(GPUResult, d_result, grid.x * sizeof(float),
             cudaMemcpyDeviceToHost);
  for (int i = 0; i < grid.x; i++) {
    gpuSum += GPUResult[i];
  }
  // kernel2: atomic function and shared memory and parallel reduction
  // copy vectors from host to device
  cudaMemcpy(d_v1, vector1, vector_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_v2, vector2, vector_size, cudaMemcpyHostToDevice);

  // launch atomic func kernel
  cudaEventRecord(start, 0);
  atomic_function_GPU_big_dot<<<grid, block>>>(d_v1, d_v2, d_result_atomic_fun,
                                               N);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime2, start, stop);
  printf("Kernel atomic_function execution time:  %3.10f sec\n",
         elapsedTime2 / 1000);
  // copy result back to host
  cudaMemcpy(GPUResult_atmoic_function, d_result_atomic_fun, sizeof(float),
             cudaMemcpyDeviceToHost);
  // kernel3 :  atomic lock and shared memory and parallel reduction
  // copy vectors from host to device
  cudaMemcpy(d_v1, vector1, vector_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_v2, vector2, vector_size, cudaMemcpyHostToDevice);

  // launch atomic lock kernel
  Lock lock;
  cudaEventRecord(start, 0);
  atomic_lock_GPU_big_dot<<<grid, block>>>(d_v1, d_v2, d_result_atomic_lock, N,
                                           lock);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime3, start, stop);
  printf("Kernel atomic_lock execution time:  %3.10f sec\n",
         elapsedTime3 / 1000);

  // copy result back to host
  cudaMemcpy(GPUResult_atmoic_lock, d_result_atomic_lock, sizeof(float),
             cudaMemcpyDeviceToHost);

  // cleanup
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_v1);
  cudaFree(d_v2);
  cudaFree(d_result);
  cudaFree(d_result_atomic_fun);
  cudaFree(d_result_atomic_lock);

  float speedup1 = elapsedTime1 / elapsedTime2;
  printf("speed up for atomic function = %.5f\t\n", speedup1);
  float speedup2 = elapsedTime1 / elapsedTime3;
  printf("speed up for atomic lock = %.5f\t\n", speedup2);

  printf("kernel1 shared memory and parallel reduction computation result = "
         "%f\t\n",
         gpuSum);
  printf("kernel2 atomic function ,shared memory and parallel reduction "
         "computation result = %f\t\n",
         GPUResult_atmoic_function);
  printf("kernel3 atomic lock ,shared memory and parallel reduction "
         "computation result = %f\t\n",
         GPUResult_atmoic_lock);

  return 0;
}
