#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <cstdio>
#include <cstdlib>

void run_solve(std::vector<std::vector<double>> &A, std::vector<double> &b, std::vector<double> &x, int num_threads, double *time)
{
    printf("Num threads: %d\n", num_threads);
    int n = b.size();
    double t = 0.0001;
    double eps = 0.000001;
    double criterion;
    //int num_iters = 0;
    double num = 0.0, denum = 0.0;

    omp_set_num_threads(num_threads);
    *time = omp_get_wtime();

    do {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            double sum = 0.0;
            for (int j = 0; j < n; ++j) {
                sum += A[i][j] * x[j];
            }
            x[i] -= t * (sum - b[i]);
        }

        num = 0.0, denum = 0.0;
        #pragma omp parallel for reduction(+:num, denum)
        for (int i = 0; i < n; ++i) {
            double sum = 0.0;
            for (int j = 0; j < n; ++j) {
                sum += A[i][j] * x[j];
            }
            double diff = sum - b[i];
            num += diff * diff;
            denum += b[i] * b[i];
        }

        criterion = std::sqrt(num) / std::sqrt(denum);
        //num_iters++;

    } while (criterion > eps);

    *time = omp_get_wtime() - *time;
    
}

void writeCSV(const char *filename, double results[15])
{
    FILE *file = fopen(filename, "w");
    if (file == NULL)
    {
        fprintf(stderr, "Error opening file for writing\n");
        exit(1);
    }

    fprintf(file, "T1, T2, S2, T4, S4, T7, S7, T8, S8, T16, S16, T20, S20, T40, S40\n");

    for (int i = 0; i < 15; ++i) {
        fprintf(file, "%.6f", results[i]);
        if (i < 14) fprintf(file, ",");
    }
    fprintf(file, "\n");

    fclose(file);
}


int main()
{
    int n = 1000;
    double time_parallel, time_serial;

    int thread_counts[8] = {1, 2, 4, 7, 8, 16, 20, 40};
    const char *filename = "results_1.csv";
    double results[15] = {0};

    std::vector<std::vector<double>> A(n, std::vector<double>(n, 1.0));
    std::vector<double> b(n, n + 1);
    std::vector<double> x(n, 0.0);

    for (int i = 0; i < n; ++i)
    {
        A[i][i] = 2.0;
    }

    run_solve(A, b, x, thread_counts[0], &time_serial);
    results[0] = time_serial;

    for (int i = 1; i < 8; ++i)
    {
        std::fill(x.begin(), x.end(), 0.0);
        run_solve(A, b, x, thread_counts[i], &time_parallel);
        results[2 * i - 1] = time_parallel;
        results[2 * i] = time_serial / time_parallel;
    }

    writeCSV(filename, results);

    return 0;
}
