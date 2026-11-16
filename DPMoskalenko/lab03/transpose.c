/*
 * Вариант C: double, N = 4000
 * Запуск: ./transpose N b threads version chunk
 * N - размер матрицы (по умолчанию 4000)
 * b - размер блока (по умолчанию 32)
 * threads - число потоков (по умолчанию все доступные)
 * version - rows, cols, blocked, rows_omp, blocked_omp,
 *             inplace, inplace_omp (по умолчанию rows)
 * chunk - для rows_omp: 0 = schedule(static), k = schedule(static, k) (по умолчанию 0)
 *
 * Пример: ./transpose 4000 32 8 blocked_omp
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define RUNS 5 // замеры для медианы
#define MIN_TIME 0.1 // минимальное время замера, чтобы не было 0.0000

void fill(double *A, int N) {
    for (int i = 0; i < N * N; i++) {
        A[i] = i;
    }
}

void copy(double *A, double *B, int N) {
    for (int i = 0; i < N * N; i++) {
        B[i] = A[i];
    }
}

// цикл по строкам A
void transpose_rows(double *A, double *B, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            B[j * N + i] = A[i * N + j];
        }
    }
}

// цикл по столбцам A
void transpose_cols(double *A, double *B, int N) {
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            B[j * N + i] = A[i * N + j];
        }
    }
}

// меньшее из двух чисел, чтобы не выходить за границы матрицы
int min_int(int x, int y) {
    if (x < y) {
        return x;
    }

    return y;
}

// блочное транспонирование
void transpose_blocked(double *A, double *B, int N, int b) {
    for (int ii = 0; ii < N; ii += b) {
        int i_end = min_int(ii + b, N);

        for (int jj = 0; jj < N; jj += b) {
            int j_end = min_int(jj + b, N);

            for (int i = ii; i < i_end; i++) {
                for (int j = jj; j < j_end; j++) {
                    B[j * N + i] = A[i * N + j];
                }
            }
        }
    }
}

// параллельная версия transpose_rows
void transpose_rows_omp(double *A, double *B, int N) {
    #pragma omp parallel for schedule(runtime)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            B[j * N + i] = A[i * N + j];
        }
    }
}

// параллельная версия transpose_blocked
void transpose_blocked_omp(double *A, double *B, int N, int b) {
    #pragma omp parallel for
    for (int ii = 0; ii < N; ii += b) {
        int i_end = min_int(ii + b, N);

        for (int jj = 0; jj < N; jj += b) {
            int j_end = min_int(jj + b, N);

            for (int i = ii; i < i_end; i++) {
                for (int j = jj; j < j_end; j++) {
                    B[j * N + i] = A[i * N + j];
                }
            }
        }
    }
}

// транспонирование на месте (относительно диагонали меняем)
void transpose_inplace(double *M, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            double tmp = M[i * N + j];
            M[i * N + j] = M[j * N + i];
            M[j * N + i] = tmp;
        }
    }
}

// параллельное транспорнирование на месте
void transpose_inplace_omp(double *M, int N) {
    #pragma omp parallel for schedule(static, 8)
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            double tmp = M[i * N + j];
            M[i * N + j] = M[j * N + i];
            M[j * N + i] = tmp;
        }
    }
}

// число несовпадений с эталоном
int mismatches(double *B, double *Ref, int N) {
    int count = 0;

    for (int i = 0; i < N * N; i++) {
        if (B[i] != Ref[i]) {
            count++;
        }
    }

    return count;
}

double median(double *t) {
    for (int i = 0; i < RUNS; i++) {
        for (int j = 0; j < RUNS - 1 - i; j++) {
            if (t[j] > t[j + 1]) {
                double tmp = t[j];
                t[j] = t[j + 1];
                t[j + 1] = tmp;
            }
        }
    }

    return t[RUNS / 2];
}

int run(char *version, double *A, double *B, int N, int b) {
    if (strcmp(version, "rows") == 0) {
        transpose_rows(A, B, N);
    } else if (strcmp(version, "cols") == 0) {
        transpose_cols(A, B, N);
    } else if (strcmp(version, "blocked") == 0) {
        transpose_blocked(A, B, N, b);
    } else if (strcmp(version, "rows_omp") == 0) {
        transpose_rows_omp(A, B, N);
    } else if (strcmp(version, "blocked_omp") == 0) {
        transpose_blocked_omp(A, B, N, b);
    } else if (strcmp(version, "inplace") == 0) {
        transpose_inplace(B, N);
    } else if (strcmp(version, "inplace_omp") == 0) {
        transpose_inplace_omp(B, N);
    } else {
        return 0;
    }

    return 1;
}

int main(int argc, char *argv[]) {
    int N = 4000;
    int b = 32;
    int threads = omp_get_max_threads();
    char *version = "rows";
    int chunk = 0;

    if (argc > 1) {
        N = atoi(argv[1]);
    }

    if (argc > 2) {
        b = atoi(argv[2]);
    }

    if (argc > 3) {
        threads = atoi(argv[3]);
    }

    if (argc > 4) {
        version = argv[4];
    }

    if (argc > 5) {
        chunk = atoi(argv[5]);
    }

    omp_set_num_threads(threads);

    // chunk = 0, то schedule(static), иначе schedule(static, chunk)
    omp_set_schedule(omp_sched_static, chunk);

    double *A = malloc(N * N * sizeof(double));
    double *B = malloc(N * N * sizeof(double));
    double *Ref = malloc(N * N * sizeof(double));

    fill(A, N);

    // эталон для проверки
    transpose_rows(A, Ref, N);

    // прогревочный запуск (заодно проверяем, что версия существует)
    if (!run(version, A, B, N, b)) {
        printf("Неизвестная версия: %s\n", version);
        return 1;
    }

    int inplace = strcmp(version, "inplace") == 0 || strcmp(version, "inplace_omp") == 0;

    double t[RUNS];
    int bad = 0;

    for (int r = 0; r < RUNS; r++) {
        copy(A, B, N);

        // повторяем транспонирование, пока замер не продлится хотя бы MIN_TIME,
        // и делим время на число повторов, получаем среднее время одного запуска
        // иначе было бы 0.0000
        int reps = 0;
        double start = omp_get_wtime();
        double elapsed;

        do {
            run(version, A, B, N, b);
            reps++;
            elapsed = omp_get_wtime() - start;
        } while (elapsed < MIN_TIME);

        t[r] = elapsed / reps;

        // транспонирование на месте, повторённое чётное число раз, ничего не делает, так что запускаем еще раз
        if (inplace && reps % 2 == 0) {
            run(version, A, B, N, b);
        }

        // проверка результата каждого запуска
        bad += mismatches(B, Ref, N);
    }

    double time = median(t);
    double bw = 2.0 * N * N * sizeof(double) / time / 1e9;

    char *check = "OK";

    if (bad > 0) {
        check = "FAIL";
    }

    printf("N=%d b=%d threads=%d version=%s chunk=%d time=%.6f s BW=%.2f GB/s check=%s\n",
           N, b, threads, version, chunk, time, bw, check);

    free(A);
    free(B);
    free(Ref);

    return 0;
}