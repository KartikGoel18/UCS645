#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>

#define N 1048576 
#define THREADS_PER_BLOCK 256

__device__ float dev_A[N];
__device__ float dev_B[N];
__device__ float dev_C[N];

__global__ void vectorAddKernel() {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        dev_C[i] = dev_A[i] + dev_B[i];
    }
}

int main() {
    float h_data[N];
    for (int i = 0; i < N; i++) h_data[i] = 1.0f;

    cudaMemcpyToSymbol(dev_A, h_data, N * sizeof(float));
    cudaMemcpyToSymbol(dev_B, h_data, N * sizeof(float));

    // Using Attribute API to avoid struct member errors
    int deviceId = 0;
    cudaGetDevice(&deviceId);
    
    int clockRateKHz, busWidthBits;
    cudaDeviceGetAttribute(&clockRateKHz, cudaDevAttrMemoryClockRate, deviceId);
    cudaDeviceGetAttribute(&busWidthBits, cudaDevAttrGlobalMemoryBusWidth, deviceId);

    // Calculation: (kHz * 1000 * bits * 2 for DDR) / conversion to GB/s
    double memClockHz = (double)clockRateKHz * 1000.0;
    double busWidthBytes = (double)busWidthBits / 8.0;
    double theoreticalBW = (memClockHz * busWidthBytes * 2.0) / 1e9;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    vectorAddKernel<<<blocks, THREADS_PER_BLOCK>>>();
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    double totalBytes = 3.0 * N * sizeof(float);
    double measuredBW = (totalBytes / 1e9) / (ms / 1000.0);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "========================================" << std::endl;
    std::cout << "        PROBLEM 3 PERFORMANCE REPORT    " << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Mem Clock Rate (MHz):  " << clockRateKHz / 1000.0 << std::endl;
    std::cout << "Mem Bus Width (bits):  " << busWidthBits << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Theoretical Bandwidth: " << theoreticalBW << " GB/s" << std::endl;
    std::cout << "Measured Bandwidth:    " << measuredBW << " GB/s" << std::endl;
    std::cout << "Kernel Execution Time: " << ms << " ms" << std::endl;
    std::cout << "Bus Efficiency:        " << (measuredBW / theoreticalBW) * 100.0 << "%" << std::endl;
    std::cout << "========================================" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}