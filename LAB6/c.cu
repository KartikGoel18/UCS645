#include <stdio.h>
#include <stdlib.h>

__global__ void matrixAdd(int *X, int *Y, int *Z, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int index = row * width + col;
        Z[index] = X[index] + Y[index];
    }
}

int main() {
    int width = 2048;
    int height = 2048;
    size_t bytes = width * height * sizeof(int);

    int *h_X = (int*)malloc(bytes);
    int *h_Y = (int*)malloc(bytes);
    int *h_Z = (int*)malloc(bytes);

    for (int i = 0; i < width * height; i++) {
        h_X[i] = 1;
        h_Y[i] = 2;
    }

    int *d_X, *d_Y, *d_Z;
    cudaMalloc(&d_X, bytes);
    cudaMalloc(&d_Y, bytes);
    cudaMalloc(&d_Z, bytes);

    cudaMemcpy(d_X, h_X, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, h_Y, bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrixAdd<<<blocksPerGrid, threadsPerBlock>>>(d_X, d_Y, d_Z, width, height);
    cudaDeviceSynchronize();

    cudaMemcpy(h_Z, d_Z, bytes, cudaMemcpyDeviceToHost);

    printf("Result at C[0]: %d\n", h_Z[0]);

    cudaFree(d_X); cudaFree(d_Y); cudaFree(d_Z);
    free(h_X); free(h_Y); free(h_Z);

    return 0;
}
