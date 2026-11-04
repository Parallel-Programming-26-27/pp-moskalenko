#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define AGG_REPS 1000   // повторы агрегирования для сравнения Gather + сумма и Reduce

// fflush(stdout) после printf - чтобы строки разных процессов не перемешивались на экране (как в прошлой лабе)

// вывод результата 
void print_result(const char *name, long long value, long long expected) {
    if (value == expected) {
        printf("%-10s sum = %lld  OK\n", name, value);
    } else {
        printf("%-10s sum = %lld  FAIL\n", name, value);
    }

    fflush(stdout);
}

int main(int argc, char **argv) {
    int rank;
    int size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 0;

    if (rank == 0 && argc > 1) {
        n = atoi(argv[1]);
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // проверка N >= P
    if (n < size) {
        if (rank == 0) {
            printf("N must be >= P = %d (got N = %d)\n", size, n);
            fflush(stdout);
        }

        MPI_Finalize();
        return 1;
    }

    // замер времени всего вычисления
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    // counts[p] - сколько элементов получает процесс p
    // displs[p] - с какого индекса начинается его часть
    // если N не делится на P, остаток раздаётся первым процессам по элементу
    int counts[size];
    int displs[size];
    int offset = 0;

    for (int p = 0; p < size; p++) {
        counts[p] = n / size;

        if (p < n % size) {
            counts[p]++;
        }

        displs[p] = offset;
        offset += counts[p];
    }

    int local_n = counts[rank];

    int *data = NULL;

    if (rank == 0) {
        data = malloc(n * sizeof(int));

        for (int i = 0; i < n; i++) {
            data[i] = i + 1;
        }
    }

    int *local = malloc(local_n * sizeof(int));

    if (n % size == 0) {
        MPI_Scatter(data, local_n, MPI_INT, local, local_n, MPI_INT, 0, MPI_COMM_WORLD);
    } else {
        // доп. задание
        // N не делится на P, части разного размера
        MPI_Scatterv(data, counts, displs, MPI_INT, local, local_n, MPI_INT, 0, MPI_COMM_WORLD);
    }

    // локальная сумма своей части; long long — чтобы не было переполнения при больших N
    long long local_sum = 0;

    for (int i = 0; i < local_n; i++) {
        local_sum += local[i];
    }

    // MPI_Gather
    // процесс 0 собирает локальные суммы в массив и складывает их
    long long all_sums[size];
    long long sum_gather = 0;

    MPI_Gather(&local_sum, 1, MPI_LONG_LONG_INT, all_sums, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int p = 0; p < size; p++) {
            sum_gather += all_sums[p];
        }
    }

    // MPI_Reduce
    // MPI сам складывает локальные суммы, результат только у процесса 0
    long long sum_reduce = 0;

    MPI_Reduce(&local_sum, &sum_reduce, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // MPI_Allreduce
    // то же сложение, но результат получают все процессы
    long long sum_allreduce = 0;

    MPI_Allreduce(&local_sum, &sum_allreduce, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);

    double time = MPI_Wtime() - start;

    // доп. задание
    // общая сумма от Allreduce есть у всех, поэтому каждый процесс считает свою долю
    printf("rank %d: elements %d..%d, local sum = %lld, Reduce = %lld, Allreduce = %lld, share = %.2f%%\n",
           rank, local[0], local[local_n - 1], local_sum, sum_reduce, sum_allreduce,
           100.0 * local_sum / sum_allreduce);
    fflush(stdout);

    // сравнение Gather + сумма и Reduce
    // одна операция слишком быстрая, поэтому повторяем AGG_REPS раз
    MPI_Barrier(MPI_COMM_WORLD);
    double t = MPI_Wtime();

    for (int k = 0; k < AGG_REPS; k++) {
        MPI_Gather(&local_sum, 1, MPI_LONG_LONG_INT, all_sums, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            sum_gather = 0;

            for (int p = 0; p < size; p++) {
                sum_gather += all_sums[p];
            }
        }
    }

    double t_gather = (MPI_Wtime() - t) / AGG_REPS;

    MPI_Barrier(MPI_COMM_WORLD);
    t = MPI_Wtime();

    for (int k = 0; k < AGG_REPS; k++) {
        MPI_Reduce(&local_sum, &sum_reduce, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    double t_reduce = (MPI_Wtime() - t) / AGG_REPS;

    if (rank == 0) {
        long long expected = (long long)n * (n + 1) / 2;

        printf("N = %d  P = %d  expected = %lld  time = %.6f s\n", n, size, expected, time);
        fflush(stdout);

        print_result("Gather", sum_gather, expected);
        print_result("Reduce", sum_reduce, expected);
        print_result("Allreduce", sum_allreduce, expected);

        printf("Gather + sum  %.3e s per operation\n", t_gather);
        printf("Reduce        %.3e s per operation\n", t_reduce);
        fflush(stdout);
    }

    free(data);
    free(local);

    MPI_Finalize();

    return 0;
}