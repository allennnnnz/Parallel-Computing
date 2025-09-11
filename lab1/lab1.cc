#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char** argv)
{
    if (argc != 3) {
        fprintf(stderr, "must provide exactly 2 arguments!\n");
        return 1;
    }
    unsigned long long r = atoll(argv[1]),
                       k = atoll(argv[2]);

    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    unsigned long long local_n  = r / size;
    unsigned long long extra    = r % size;
    unsigned long long exactly_n = local_n + (rank < extra ? 1 : 0);
    unsigned long long start     = rank * local_n + (rank < extra ? rank : extra);

    unsigned long long y = 0;
    for (unsigned long long i = start; i < start + exactly_n; i++) {
        y += (unsigned long long) ceil(sqrtl(r * r - i * i));
		y %= k;
		
    }

    unsigned long long global_sum = 0;
    MPI_Reduce(&y, &global_sum, 1,
               MPI_UNSIGNED_LONG_LONG, MPI_SUM,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("%llu\n", (4 * global_sum) % k);
    }

    MPI_Finalize();
    return 0;
}