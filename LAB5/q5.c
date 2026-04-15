#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int is_perfect(int n) {
    if (n <= 1) return 0;
    int sum = 1;
    for (int i = 2; i * i <= n; i++) {
        if (n % i == 0) {
            sum += i;
            if (i * i != n) {
                sum += n / i;
            }
        }
    }
    return (sum == n);
}

int main(int argc, char** argv) {
    int rank, size;
    int MAX_VAL = 10000; 
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) { 
        double start_time = MPI_Wtime();
        int next_num = 2;
        int active_workers = size - 1;
        int response;
        MPI_Status status;
        int perfect_count = 0;

        while (active_workers > 0) {
            MPI_Recv(&response, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
            int slave_id = status.MPI_SOURCE;

            if (response > 0) {
                perfect_count++;
            }

            if (next_num <= MAX_VAL) {
                MPI_Send(&next_num, 1, MPI_INT, slave_id, 0, MPI_COMM_WORLD);
                next_num++;
            } else {
                int terminate = -1;
                MPI_Send(&terminate, 1, MPI_INT, slave_id, 0, MPI_COMM_WORLD);
                active_workers--;
            }
        }
        double end_time = MPI_Wtime();
        printf("--- Perfect Number Search (P=%d) ---\n", size);
        printf("Perfect Numbers Found: %d\n", perfect_count);
        printf("Total Time: %f seconds\n\n", end_time - start_time);
        
    } else {
        int request = 0; 
        int num_to_test;
        while (1) {
            MPI_Send(&request, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Recv(&num_to_test, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (num_to_test == -1) break; 
            
            if (is_perfect(num_to_test)) {
                request = num_to_test;
            } else {
                request = -num_to_test;
            }
        }
    }

    MPI_Finalize();
    return 0;
}