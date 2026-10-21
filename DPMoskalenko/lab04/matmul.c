/*
 * запуск: ./matmul N b threads
 * N - размер матрицы (по умолчанию 512)
 * b - размер блока для blocked_omp (по умолчанию 32)
 * threads - число потоков (по умолчанию все доступные)
 *
 * Пример: ./matmul 1024 32 8
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define RUNS 5  // число замеров для медианы

// заполнение матриц воспроизводимыми данными
void fill(double *A, double *B, int N) {
    for (int i = 0; i < N * N; i++) {
        A[i] = (i % 10) * 0.1;
        B[i] = (i % 7) * 0.1;
    }
}

// сброс матрицы результата
void reset(double *C, int N) {
    for (int i = 0; i < N * N; i++) {
        C[i] = 0.0;
    }
}

// последовательное умножение, порядок циклов i, j, k
void ijk_seq(double *A, double *B, double *C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;

            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }

            C[i * N + j] = sum;
        }
    }
}

// параллельное умножение, порядок i, j, k
void ijk_omp(double *A, double *B, double *C, int N) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;

            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }

            C[i * N + j] = sum;
        }
    }
}

// последовательное умножение, порядок циклов i, k, j
void ikj_seq(double *A, double *B, double *C, int N) {
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            double a = A[i * N + k];

            for (int j = 0; j < N; j++) {
                C[i * N + j] += a * B[k * N + j];
            }
        }
    }
}

// параллельное умножение, порядок i, k, j
void ikj_omp(double *A, double *B, double *C, int N) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            double a = A[i * N + k];

            for (int j = 0; j < N; j++) {
                C[i * N + j] += a * B[k * N + j];
            }
        }
    }
}

// чтобы не выходить за блок
int min_int(int x, int y) {
    if (x < y) {
        return x;
    }

    return y;
}

// блочное умножение
void blocked_omp(double *A, double *B, double *C, int N, int b) {
    #pragma omp parallel for
    for (int ii = 0; ii < N; ii += b) {
        int i_end = min_int(ii + b, N);

        for (int kk = 0; kk < N; kk += b) {
            int k_end = min_int(kk + b, N);

            for (int jj = 0; jj < N; jj += b) {
                int j_end = min_int(jj + b, N);

                // умножение блоков, внутри порядок i, k, j
                for (int i = ii; i < i_end; i++) {
                    for (int k = kk; k < k_end; k++) {
                        double a = A[i * N + k];

                        for (int j = jj; j < j_end; j++) {
                            C[i * N + j] += a * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

// максимальное отличие матрицы C от эталона Ref
double max_diff(double *C, double *Ref, int N) {
    double m = 0.0;

    for (int i = 0; i < N * N; i++) {
        double d = fabs(C[i] - Ref[i]);

        if (d > m) {
            m = d;
        }
    }

    return m;
}

// медиана из RUNS замеров
double median(double *t) {
    // сортировка пузырьком
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

// запуск разных варинатов
void run(int v, double *A, double *B, double *C, int N, int b) {
    switch (v) {
        case 0:
            ijk_seq(A, B, C, N);
            break;
        case 1:
            ijk_omp(A, B, C, N);
            break;
        case 2:
            ikj_seq(A, B, C, N);
            break;
        case 3:
            ikj_omp(A, B, C, N);
            break;
        case 4:
            blocked_omp(A, B, C, N, b);
            break;
    }
}

double measure(int v, double *A, double *B, double *C, int N, int b) {
    double t[RUNS];

    for (int r = 0; r < RUNS; r++) {
        reset(C, N);

        double start = omp_get_wtime();
        run(v, A, B, C, N, b);
        t[r] = omp_get_wtime() - start;
    }

    return median(t);
}

int main(int argc, char *argv[]) {
    int N = 512;
    int b = 32;
    int threads = omp_get_max_threads();

    if (argc > 1) {
        N = atoi(argv[1]);
    }

    if (argc > 2) {
        b = atoi(argv[2]);
    }

    if (argc > 3) {
        threads = atoi(argv[3]);
    }

    omp_set_num_threads(threads);

    double *A = malloc(N * N * sizeof(double));
    double *B = malloc(N * N * sizeof(double));
    double *C = malloc(N * N * sizeof(double));
    double *Ref = malloc(N * N * sizeof(double));

    fill(A, B, N);

    printf("N = %d  b = %d  threads = %d  runs = %d\n", N, b, threads, RUNS);

    const char *names[] = {"ijk_seq", "ijk_omp", "ikj_seq", "ikj_omp", "blocked_omp"};
    int count = 5;

    for (int v = 0; v < count; v++) {
        double t = measure(v, A, B, C, N, b);

        // результат ijk_seq сохраняем как эталон
        if (v == 0) {
            for (int i = 0; i < N * N; i++) {
                Ref[i] = C[i];
            }
        }

        double diff = max_diff(C, Ref, N);

        if (diff < 1e-6) {
            printf("%-12s time = %.4f s  check = OK (max diff = %.1e)\n", names[v], t, diff);
        } else {
            printf("%-12s time = %.4f s  check = FAIL (max diff = %.1e)\n", names[v], t, diff);
        }
    }

    free(A);
    free(B);
    free(C);
    free(Ref);

    return 0;
}