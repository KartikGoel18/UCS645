#include <stdio.h>

int main() {
    int numGPUs;
    cudaGetDeviceCount(&numGPUs);

    if (numGPUs == 0) {
        printf("Error: Zero CUDA-capable GPUs detected on this system.\n");
        return 0;
    }

    for (int gpuIdx = 0; gpuIdx < numGPUs; ++gpuIdx) {
        cudaDeviceProp gpuProps;
        cudaGetDeviceProperties(&gpuProps, gpuIdx);

        printf("--- GPU ID %d: %s ---\n", gpuIdx, gpuProps.name);
        printf("  Architecture Version:        %d.%d\n", gpuProps.major, gpuProps.minor);
        printf("  Max Threads per Block (xyz): (%d, %d, %d)\n", gpuProps.maxThreadsDim[0],
                                                                gpuProps.maxThreadsDim[1], gpuProps.maxThreadsDim[2]);
        printf("  Max Grid Size (xyz):         (%d, %d, %d)\n", gpuProps.maxGridSize[0],
                                                                gpuProps.maxGridSize[1], gpuProps.maxGridSize[2]);
        printf("  Thread Limit per Block:      %d\n", gpuProps.maxThreadsPerBlock);
        printf("  Global VRAM Capacity:        %.2f GB\n", (float)gpuProps.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Shared Mem per Block:        %lu B\n", gpuProps.sharedMemPerBlock);
        printf("  Constant Memory Pool:        %lu B\n", gpuProps.totalConstMem);
        printf("  Warp Thread Count:           %d\n", gpuProps.warpSize);
        printf("  Supports Concurrent Kernels: %s\n\n", gpuProps.concurrentKernels ? "True" : "False");
    }
    return 0;
}