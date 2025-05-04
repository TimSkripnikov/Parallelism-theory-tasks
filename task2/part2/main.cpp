#include <stdio.h>
#include <stdlib.h>
#include <math.h> // Для exp, sqrt, fabs
#include <omp.h>  // Для OpenMP


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
        double sumloc = 0.0;

        #pragma omp for
        for (int i = 0; i < n; i++)
            sumloc += func(a + h * (i + 0.5));

        #pragma omp atomic
        sum += sumloc;
    
    }
    sum *= h;
    return sum;
}


void run_serial(double *time)
{
    *time = omp_get_wtime();
    double res = integrate(a, b, nsteps);
    *time = omp_get_wtime() - *time;
    printf("Result (serial): %.12f; error %.12f\n", res, fabs(res - sqrt(PI)));
}

void run_parallel(int num_threads, double *time)
{
    omp_set_num_threads(num_threads); 
    printf("Running parallel version with %d threads...\n", num_threads);

    *time = omp_get_wtime();
    double res = integrate_omp(func, a, b, nsteps);
    *time = omp_get_wtime() - *time;
    printf("Result (parallel, %d threads): %.12f; error %.12f\n", num_threads, res, fabs(res - sqrt(PI)));
}

int main()
{
    double time_serial, time_parallel;
    int threads[] = {1, 2, 4, 7, 8, 16, 20, 40};

    run_serial(&time_serial);

    FILE *file = fopen("results.csv", "w");
    if (!file)
    {
        perror("Error opening file");
        return 1;
    }

    fprintf(file, "Threads,Time,Speedup\n");

    for (int i = 0; i < 8; i++)
    {
        run_parallel(threads[i], &time_parallel);
        fprintf(file, "%d,%.6f,%.2f\n", threads[i], time_parallel, time_serial/time_parallel);
    }

    fclose(file);
    printf("Results saved to results.csv\n");

    return 0;
}
