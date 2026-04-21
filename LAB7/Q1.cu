#include <iostream>
#include <cuda_runtime.h>

__global__ void taskKernel(int n, long long* result) {
    int tid = threadIdx.x;

    // Task A: Iterative approach (O(n))
    if (tid == 0) {
        long long sum = 0;
        for (int i = 1; i <= n; i++) sum += i;
        result[0] = sum;
    }
    // Task B: Direct formula (O(1))
    else if (tid == 1) {
        result[1] = (1LL * n * (n + 1)) / 2;
    }
}

int main() {
    const int N = 1024;
    long long *d_res;
    long long h_res[2];
    cudaMalloc(&d_res, 2 * sizeof(long long));

    taskKernel<<<1, 32>>>(N, d_res);
    cudaDeviceSynchronize();

    cudaMemcpy(h_res, d_res, 2 * sizeof(long long), cudaMemcpyDeviceToHost);
    printf("Thread 0 (Iterative): %lld\nThread 1 (Formula): %lld\n", h_res[0], h_res[1]);

    cudaFree(d_res);
    return 0;
}
