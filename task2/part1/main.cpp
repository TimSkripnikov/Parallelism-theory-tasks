#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void matrix_vector_product(double *a, double *b, double *c, int m, int n){
    for (int i = 0; i < m; i++){
        c[i] = 0.0;
        for (int j = 0; j < n; j++){
            c[i] += a[i * n + j] * b[j];
        }
    }
}

void run_serial(int m, int n, double *time){
    double *a, *b, *c;

    a = (double *)malloc(sizeof(*a) * m * n);
    b = (double *)malloc(sizeof(*b) * n);
    c = (double *)malloc(sizeof(*c) * m);

    for (int i = 0; i < m; ++i){
        for (int j = 0; j < n; ++j){
            a[i * n +j] = i + j;
        }
    }

    for (int j = 0; j < n; ++j)
        b[j] = j;

    *time = omp_get_wtime();
    matrix_vector_product(a, b, c, m, n);
    *time = omp_get_wtime() - *time;

    free(a);
    free(b);
    free(c);
}

void matrix_vector_product_omp(double *a, double *b, double *c, int m, int n){
    
    #pragma omp parallel for
    
    // int num_threads = omp_get_num_threads();
    // int threadid = omp_get_thread_num();
    // int items_per_thread = m / num_threads;

    // int lower_bound = threadid * items_per_thread;
    // int upper_bound = (threadid == num_threads - 1) ? (m - 1) : (lower_bound + items_per_thread - 1);
    
    for (int i = 0; i < m; ++i){
        c[i] = 0.0;
        for (int j = 0; j < n; ++j){
            c[i] += a[i * n + j] * b[j];
        }
    }
    
}


void run_parallel(int m, int n, int num_threads, double *time)
{
    double *a, *b, *c;

    a = (double *)malloc(sizeof(*a) * m * n);
    b = (double *)malloc(sizeof(*b) * n);
    c = (double *)malloc(sizeof(*c) * m);
    
    #pragma omp parallel for

    for (int i = 0; i < m; ++i){
        for (int j = 0; j < n; ++j){
            a[i * n + j] = i + j;
        }
    }

    #pragma omp parallel for

    for (int j = 0; j < n; ++j){
        b[j] = j;
    }


    omp_set_num_threads(num_threads);
    *time = omp_get_wtime();
    matrix_vector_product_omp(a, b, c, m, n);
    *time = omp_get_wtime() - *time;

    free(a);
    free(b);
    free(c);

}


    void writeCSV (const char *filename, double results[2][16])
    {
        FILE *file = fopen(filename, "w");
        if (file == NULL)
        {
            fprintf(stderr, "Error opening file for writing\n");
            exit(1);
        }

        fprintf(file, "N=M, T1, T2, S2, T4, S4, T7, S7, T8, S8, T16, S16, T20, S20, T40, S40\n");

        for (int str = 0; str < 2; ++str)
        {
            fprintf(file, "%d", (str == 0) ? 20000 : 40000);
            for (int column = 0; column < 15; ++column) {
                fprintf(file, ",%.6f", results[str][column]);
            }
            fprintf(file, "\n");
        }

        fclose(file);

    }

int main(){
    int thread_counts[7] = {2, 4, 7, 8, 16, 20, 40};
    const char *filename = "results.csv";
    double results[2][16] = {0};

    int m, n;
    double time_serial, time_parallel;

    for (int test = 0; test < 2; test++)
    {
        
    m = (test == 0 ) ? 20000 : 40000;
    n = m;

    run_serial(m, n, &time_serial);
    results[test][0] = time_serial;

    for (int i = 0; i < 7; ++i)
    {
        run_parallel(m, n, thread_counts[i], &time_parallel);
        results[test][2 * i + 1] = time_parallel;
        results[test][2 * i + 2] = time_serial / time_parallel;
    }

    }


    writeCSV(filename, results);

    printf("All is ok!\n");

    return 0;

}