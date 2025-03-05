#include <stdio.h>
#include <stdlib.h>
#include <math.h> // Для exp, sqrt, fabs
#include <omp.h>  // Для OpenMP
#include <time.h> // Для измерения времени

const double PI = 3.14159265358979323846;
const double a = -4.0;
const double b = 4.0;
const int nsteps = 40000000;

double func(double x)
{
    return exp(-x * x);
}

double integrate(double a, double b, int n)
{
    double h = (b - a) / n;
    double sum = 0.0;
    for (int i = 0; i < n; i++)
        sum += func(a + h * (i + 0.5));
    sum *= h;
    return sum;
}

double integrate_omp(double (*func)(double), double a, double b, int n)
{
    double h = (b - a) / n;
    double sum = 0.0;

#pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = n / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (n - 1) : (lb + items_per_thread - 1);

        double sumloc = 0.0;
        for (int i = lb; i <= ub; i++)
            sumloc += func(a + h * (i + 0.5));

#pragma omp atomic
        sum += sumloc;
    }

    sum *= h;
    return sum;
}

// Функция для измерения времени работы
double wtime()
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

double run_serial()
{
    double t = wtime();
    double res = integrate(a, b, nsteps);
    t = wtime() - t;
    printf("Result (serial): %.12f; error %.12f\n", res, fabs(res - sqrt(PI)));
    return t;
}

double run_parallel(int num_threads)
{
    omp_set_num_threads(num_threads); // Устанавливаем количество потоков
    printf("Running parallel version with %d threads...\n", num_threads);

    double t = wtime();
    double res = integrate_omp(func, a, b, nsteps);
    t = wtime() - t;
    printf("Result (parallel, %d threads): %.12f; error %.12f\n", num_threads, res, fabs(res - sqrt(PI)));
    return t;
}

int main(int argc, char **argv)
{
    printf("Integration f(x) on [%.12f, %.12f], nsteps = %d\n", a, b, nsteps);

    // Запуск последовательной версии
    double tserial = run_serial();

    // Массив с количеством потоков
    int threads[] = {1, 2, 4, 7, 8, 16, 20, 40};
    int num_tests = sizeof(threads) / sizeof(threads[0]);

    // Открываем CSV файл для записи результатов
    FILE *file = fopen("results.csv", "w");
    if (!file)
    {
        perror("Error opening file");
        return 1;
    }

    // Записываем заголовок
    fprintf(file, "Threads,Time,Speedup\n");

    // Запускаем параллельные версии и записываем данные
    for (int i = 0; i < num_tests; i++)
    {
        int num_threads = threads[i];
        double tparallel = run_parallel(num_threads);
        double speedup = tserial / tparallel;
        printf("Speedup with %d threads: %.2f\n", num_threads, speedup);

        // Записываем в CSV
        fprintf(file, "%d,%.6f,%.2f\n", num_threads, tparallel, speedup);
    }

    fclose(file);
    printf("Results saved to results.csv\n");

    return 0;
}
