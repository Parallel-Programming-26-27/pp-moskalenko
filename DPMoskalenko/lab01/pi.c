#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

double pi_seq(long long n) {
    double sum = 0.0;
    for (long long i = 0; i < n; i++) {
        double term = 1.0 / (2.0 * i + 1.0);
        if (i % 2 == 0) {
            sum += term;
        } else {
            sum -= term;
        }
    }
    return 4.0 * sum;
}

double pi_par(long long n) {
    double sum = 0.0;

    #pragma omp parallel for reduction(+:sum)
    for (long long i = 0; i < n; i++) {
        double term = 1.0 / (2.0 * i + 1.0);

        if (i % 2 == 0) {
            sum += term;
        } else {
            sum -= term;
        }
    }

    return 4.0 * sum;
}

int main(int argc, char *argv[]) {
    long long n = 1000000000LL;

    if (argc > 1) {
        n = atoll(argv[1]);
    }

    printf("n = %lld\n", n);

    // последовательная версия
    double t_seq = omp_get_wtime();
    double pi1 = pi_seq(n);
    t_seq = omp_get_wtime() - t_seq;

    printf("[seq] pi = %.12f  error = %.3e  time = %.3f s  threads = 1\n",
           pi1, fabs(pi1 - M_PI), t_seq);

    // параллельная версия
    int threads = omp_get_max_threads();

    double t_par = omp_get_wtime();
    double pi2 = pi_par(n);
    t_par = omp_get_wtime() - t_par;

    printf("[par] pi = %.12f  error = %.3e  time = %.3f s  threads = %d\n",
           pi2, fabs(pi2 - M_PI), t_par, threads);

    // ускорение
    printf("speedup = %.2f\n", t_seq / t_par);

    return 0;
}