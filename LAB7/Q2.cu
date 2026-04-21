#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cuda_runtime.h>

#define MIN(a,b) (((a)<(b))?(a):(b))

// --- CPU Merge Logic ---
void merge(int arr[], int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;
    std::vector<int> L(n1), R(n2);
    for (int i = 0; i < n1; i++) L[i] = arr[l + i];
    for (int j = 0; j < n2; j++) R[j] = arr[m + 1 + j];
    int i = 0, j = 0, k = l;
    while (i < n1 && j < n2) arr[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];
    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];
}

void mergeSortCPU(int arr[], int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;
        mergeSortCPU(arr, l, m);
        mergeSortCPU(arr, m + 1, r);
        merge(arr, l, m, r);
    }
}

// --- GPU Merge Logic (Iterative) ---
__device__ void gpu_merge(int* arr, int* temp, int l, int m, int r) {
    int i = l, j = m, k = l;
    while (i < m && j < r) temp[k++] = (arr[i] <= arr[j]) ? arr[i++] : arr[j++];
    while (i < m) temp[k++] = arr[i++];
    while (j < r) temp[k++] = arr[j++];
    for (i = l; i < r; i++) arr[i] = temp[i];
}

__global__ void mergeSortKernel(int* arr, int* temp, int n, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int l = idx * width * 2;
    if (l < n) {
        int m = MIN(l + width, n);
        int r = MIN(l + 2 * width, n);
        gpu_merge(arr, temp, l, m, r);
    }
}

int main() {
    const int n = 1000;
    int h_arr[n], h_gpu_orig[n];
    for (int i = 0; i < n; i++) {
        h_arr[i] = rand() % 1000;
        h_gpu_orig[i] = h_arr[i]; // Keep a copy for GPU
    }

    // 1. CPU Sort Timing
    auto s1 = std::chrono::high_resolution_clock::now();
    mergeSortCPU(h_arr, 0, n - 1);
    auto e1 = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(e1 - s1).count();

    // 2. GPU Sort Timing
    int *d_arr, *d_temp;
    cudaMalloc(&d_arr, n * sizeof(int));
    cudaMalloc(&d_temp, n * sizeof(int));
    cudaMemcpy(d_arr, h_gpu_orig, n * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int width = 1; width < n; width *= 2) {
        int threads = 128;
        int blocks = (n / (2 * width) + threads - 1) / threads;
        mergeSortKernel<<<blocks, threads>>>(d_arr, d_temp, n, width);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);

    printf("--- PROBLEM 2 PERFORMANCE ---\n");
    printf("CPU Merge Sort Time: %f ms\n", cpu_time);
    printf("GPU Merge Sort Time: %f ms\n", gpu_time);
    printf("Speedup:             %f x\n", cpu_time / gpu_time);

    cudaFree(d_arr); cudaFree(d_temp);
    return 0;
}