    #include <stdio.h>
    #include <stdlib.h>

    __global__ void arraySum(float *d_in, float *d_out, int size) {
        extern __shared__ float sdata[];

        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

        sdata[tid] = (i < size) ? d_in[i] : 0.0f;
        __syncthreads();

        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            atomicAdd(d_out, sdata[0]);
        }
    }

    int main() {
        int N = 100000;
        size_t bytes = N * sizeof(float);

        float *h_in = (float*)malloc(bytes);
        float h_out = 0.0f;
        for (int i = 0; i < N; i++) {
            h_in[i] = 1.0f;
        }

        float *d_in, *d_out;
        cudaMalloc(&d_in, bytes);
        cudaMalloc(&d_out, sizeof(float));

        cudaMemset(d_out, 0, sizeof(float));

        cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

        int threadsPerBlock = 256;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        int sharedMemSize = threadsPerBlock * sizeof(float);

        arraySum<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_in, d_out, N);

        cudaDeviceSynchronize();

        cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

        printf("Sum is: %f\n", h_out);

        cudaFree(d_in);
        cudaFree(d_out);
        free(h_in);

        return 0;
    }
