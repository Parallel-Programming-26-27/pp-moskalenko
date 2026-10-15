#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// вариант E
double f(double x) {
    return sqrt(exp(x) - 1.0);
}

// последовательно
double integral_seq(double a, double b, long long n) {
    double h = (b - a) / n;
    double sum = 0.0;

    for (long long i = 0; i < n; i++) {
        double x = a + (i + 0.5) * h;
        sum += f(x);
    }

    return h * sum;
}

// параллельно
double integral_par(double a, double b, long long n) {
    double h = (b - a) / n;
    double sum = 0.0;

    #pragma omp parallel for reduction(+:sum)
    for (long long i = 0; i < n; i++) {
        double x = a + (i + 0.5) * h;
        sum += f(x);
    }

    return h * sum;
}

// удваиваем n, пока разность интегралов >= eps
// parallel = 0 - последовательная версия, 1 - параллельная
double integral_eps(double a, double b, long long n, double eps, int parallel, long long *n_found) {
    double prev;
    double curr;

    if (parallel) {
        prev = integral_par(a, b, n);
    } else {
        prev = integral_seq(a, b, n);
    }

    while (1) {
        n = n * 2;

        if (parallel) {
            curr = integral_par(a, b, n);
        } else {
            curr = integral_seq(a, b, n);
        }

        if (fabs(curr - prev) < eps) {
            break;
        }

        prev = curr;
    }

    *n_found = n;
    return curr;
}

int main(int argc, char *argv[]) {
    double a = 0.1;
    double b = 2.0;
    long long n = 100000000LL;
    int threads = omp_get_max_threads();

    if (argc > 1) {
        a = atof(argv[1]);
    }

    if (argc > 2) {
        b = atof(argv[2]);
    }

    if (argc > 3) {
        n = atoll(argv[3]);
    }

    if (argc > 4) {
        threads = atoi(argv[4]);
    }

    omp_set_num_threads(threads);

    if (argc > 5) {
        double eps = atof(argv[5]);
        long long n_seq;
        long long n_par;

        printf("a = %g  b = %g  start n = %lld  eps = %g\n", a, b, n, eps);

        // последовательно
        double t_seq = omp_get_wtime();
        double res_seq = integral_eps(a, b, n, eps, 0, &n_seq);
        t_seq = omp_get_wtime() - t_seq;

        printf("[seq] I = %.12f  n = %lld  time = %.3f s  threads = 1\n", res_seq, n_seq, t_seq);
        
        // параллельно
        double t_par = omp_get_wtime();
        double res_par = integral_eps(a, b, n, eps, 1, &n_par);
        t_par = omp_get_wtime() - t_par;

        printf("[par] I = %.12f  n = %lld  time = %.3f s  threads = %d\n", res_par, n_par, t_par, threads);

        return 0;
    }

    printf("a = %g  b = %g  n = %lld\n", a, b, n);

    // последовательно
    double t_seq = omp_get_wtime();
    double res_seq = integral_seq(a, b, n);
    t_seq = omp_get_wtime() - t_seq;

    printf("[seq] I = %.12f  time = %.3f s  threads = 1\n", res_seq, t_seq);

    // паралелльно
    double t_par = omp_get_wtime();
    double res_par = integral_par(a, b, n);
    t_par = omp_get_wtime() - t_par;

    printf("[par] I = %.12f  time = %.3f s  threads = %d\n", res_par, t_par, threads);
    
    printf("speedup = %.2f\n", t_seq / t_par);

    return 0;
}