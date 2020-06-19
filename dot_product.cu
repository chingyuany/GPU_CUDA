#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define N 1024 * 1024
#define threads_per_block 512

__global__ void GPU_big_dot(float *a, float *b, float *c, int n) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n)
    c[index] = a[index] * b[index];
}

long long start_timer() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec * 1000000 + tv.tv_usec;
}
long long stop_timer(long long start_time, const char *name) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  long long end_time = tv.tv_sec * 1000000 + tv.tv_usec;
  printf("%s: %.5f sec\n", name,
         ((float)(end_time - start_time)) / (1000 * 1000));
  return end_time - start_time;
}

float *CPU_big_dot(float *A, float *B, int n) {
  static float cpuResult = 0.0;
  for (int i = 0; i < n; i++) {
    // printf("\n a * b = %f\t\n", A[i] * B[i]);
    cpuResult += A[i] * B[i];
  }
  // printf("inside cpu function's sum = %f\t\n", cpuResult);
  return &cpuResult;
}
float *GPU_big_dot(float *vector1, float *vector2, int vector_size) {
  float *d_v1, *d_v2, *d_result, *GPUResult;
  static float gpuSum = 0.0;
  // allocate memory space for device memory
  cudaMalloc((void **)&d_v1, vector_size);
  cudaMalloc((void **)&d_v2, vector_size);
  cudaMalloc((void **)&d_result, vector_size);
  GPUResult = (float *)malloc(vector_size);

  // copy vectors from host to device
  long long GPU_start_time = start_timer();
  cudaMemcpy(d_v1, vector1, vector_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_v2, vector2, vector_size, cudaMemcpyHostToDevice);
  long long GPU_total_time = stop_timer(
      GPU_start_time,
      (char *)"Memory allocation and data transfer from CPU to GPU time ");

  // launch add()kernel
  GPU_start_time = start_timer();
  GPU_big_dot<<<(N + threads_per_block - 1) / threads_per_block,
                threads_per_block>>>(d_v1, d_v2, d_result, N);
  GPU_total_time = stop_timer(GPU_start_time, (char *)"Kernel execution time");

  // copy result back to host
  GPU_start_time = start_timer();
  cudaMemcpy(GPUResult, d_result, vector_size, cudaMemcpyDeviceToHost);
  GPU_total_time =
      stop_timer(GPU_start_time, (char *)"Data transfer from GPU to CPU time");
  // cleanup
  cudaFree(d_v1);
  cudaFree(d_v2);
  cudaFree(d_result);

  for (int i = 0; i < N; i++) {
    gpuSum += GPUResult[i];
    // printf("mutlix %f\t\n", GPUResult[i]);
  }
  return &gpuSum;
  // printf("gpuSum in function %f\t\n", gpuSum);
}

void random_vecotr_init(float *vector) {

  for (int i = 0; i < N; i++) {
    vector[i] = (float)rand() / RAND_MAX;
  }
}
void print_vector(float *vector, char *name) {
  printf("%s = \n", name);
  for (int i = 0; i < N; i++) {
    printf("%f\t\n", vector[i]);
  }
}
int main(int argc, char *argv[]) {
  float *vector1, *vector2, *finalCPUResult, *finalGPUResult;
  int vector_size = sizeof(float) * N;

  // allocate memory space for host memory
  vector1 = (float *)malloc(vector_size);
  vector2 = (float *)malloc(vector_size);

  // init two vectors with random float numbers
  srand(time(NULL));
  random_vecotr_init(vector1);
  random_vecotr_init(vector2);
  // print_vector(vector1, (char *)"vector1");
  // print_vector(vector2, (char *)"vector2");

  // run on cpu
  long long cpu_start_time = start_timer();
  finalCPUResult = CPU_big_dot(vector1, vector2, N);
  long long cpu_total_time = stop_timer(
      cpu_start_time, (char *)"Total computation time for CPU_big_dot()");

  // run on gpu
  long long GPU_total_start_time = start_timer();
  finalGPUResult = GPU_big_dot(vector1, vector2, vector_size);
  long long GPU_total_end_time = stop_timer(
      GPU_total_start_time, (char *)"Total computation time for GPU_big_dot()");

  // convert to seconds
  float cpu_total_time_sec = (float)cpu_total_time / (1000 * 1000);
  float GPU_total_end_time_sec = (float)GPU_total_end_time / (1000 * 1000);

  float speedup = cpu_total_time_sec / GPU_total_end_time_sec;
  printf("speed up = %.5f\t\n", speedup);
  float compareResult = *finalCPUResult - *finalGPUResult;
  if (fabs(compareResult) <= 0.000001) {
    printf("CPU computing result and GPU computing result are the same, the "
           "results are correct, the difference is %f\n",
           compareResult);
  } else {
    printf("CPU computing result and GPU computing result are NOT the same, "
           "the results are NOT correct, the difference is %f\n",
           compareResult);
  }

  printf("CPU computation result in float format = %f\t\n", *finalCPUResult);
  printf("CPU computation result in  scientific notation= %e\t\n",
         *finalCPUResult);
  printf("GPU computation result in float format =  %f\t\n", *finalGPUResult);
  printf("GPU computation result in  scientific notation = %e\t\n",
         *finalGPUResult);
  return 0;
}
