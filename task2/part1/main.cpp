#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void *xmalloc(size_t size)
{
    void *ptr = malloc(size);
    if (ptr == NULL)
    {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    return ptr;
}

double wtime()
{
    return omp_get_wtime();
}

void matrix_vector_product(double *a, double *b, double *c, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        c[i] = 0.0;
        for (int j = 0; j < n; j++)
            c[i] += a[i * n + j] * b[j];
    }
}

void matrix_vector_product_omp(double *a, double *b, double *c, int m, int n)
{
#pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = m / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (m - 1) : (lb + items_per_thread - 1);

        for (int i = lb; i <= ub; i++)
        {
            c[i] = 0.0;
            for (int j = 0; j < n; j++)
                c[i] += a[i * n + j] * b[j];
        }
    }
}

void run_serial(int m, int n, double *t_serial)
{
    double *a, *b, *c;
    a = (double *)xmalloc(sizeof(*a) * m * n);
    b = (double *)xmalloc(sizeof(*b) * n);
    c = (double *)xmalloc(sizeof(*c) * m);

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
            a[i * n + j] = i + j;
    }

    for (int j = 0; j < n; j++)
        b[j] = j;

    double t = wtime();
    matrix_vector_product(a, b, c, m, n);
    t = wtime() - t;

    *t_serial = t;

    free(a);
    free(b);
    free(c);
}

void run_parallel(int m, int n, int num_threads, double *t_parallel)
{
    double *a, *b, *c;
    a = (double *)xmalloc(sizeof(*a) * m * n);
    b = (double *)xmalloc(sizeof(*b) * n);
    c = (double *)xmalloc(sizeof(*c) * m);

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
            a[i * n + j] = i + j;
    }

    for (int j = 0; j < n; j++)
        b[j] = j;

    omp_set_num_threads(num_threads);
    double t = wtime();
    matrix_vector_product_omp(a, b, c, m, n);
    t = wtime() - t;

    *t_parallel = t;

    free(a);
    free(b);
    free(c);
}

void write_to_csv(const char *filename, double results[2][16])
{
    FILE *file = fopen(filename, "w");
    if (file == NULL)
    {
        fprintf(stderr, "Error opening file for writing\n");
        exit(1);
    }

    // Заголовок
    fprintf(file, "N=M, T1, T2, S2, T4, S4, T7, S7, T8, S8, T16, S16, T20, S20, T40, S40\n");

    // Две строки с результатами
    for (int i = 0; i < 2; i++)
    {
        fprintf(file, "%d", (i == 0) ? 20000 : 40000); // N=M
        for (int j = 0; j < 15; j++)
        {
            fprintf(file, ", %.6f", results[i][j]); // Значения
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

int main()
{
    int thread_counts[7] = {2, 4, 7, 8, 16, 20, 40}; // Количество потоков для тестирования
    const char *filename = "results.csv";
    double results[2][16] = {0}; // Хранит результаты для 20000 и 40000

    for (int test = 0; test < 2; test++)
    {
        int m = (test == 0) ? 20000 : 40000;
        int n = m;
        double t_serial;

        // Последовательный запуск
        run_serial(m, n, &t_serial);
        results[test][0] = t_serial; // T1

        for (int i = 0; i < 7; i++)
        {
            double t_parallel;
            run_parallel(m, n, thread_counts[i], &t_parallel);
            results[test][2 * i + 1] = t_parallel;            // T (с разным числом потоков)
            results[test][2 * i + 2] = t_serial / t_parallel; // S (ускорение)
        }
    }

    write_to_csv(filename, results);
    return 0;
}
