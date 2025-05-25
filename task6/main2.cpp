#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>

#define OUT_FILE "result.dat"

#define TAU 0.25
#define EPS 1e-6
#define MAX_ITER 10000

#ifndef NX
#define NX 128
#endif
#ifndef NY
#define NY NX
#endif
#define SIZE (NX * NY)

using namespace std::chrono;

const double corner_values[4] = {10.0, 20.0, 30.0, 20.0};

double get_b(int idx) {
    if (idx == NY/2*NX + NX/3) return 10;
    if (idx == NY*2/3*NX + NX*2/3) return -25;
    return 0;
}

double get_a(int row, int col) {
    if (row == col) return 4.0;
    if (col == row - 1 && row % NX != 0)    return -1.0;
    if (col == row + 1 && (row + 1) % NX != 0) return -1.0;
    if (col == row - NX && row >= NX)       return -1.0;
    if (col == row + NX && row < SIZE - NX) return -1.0;
    return 0.0;
}

void init_b(double *b) {
    #pragma acc enter data create(b[0:SIZE])
    
    #pragma acc parallel loop present(b)
    for (int i = 0; i < SIZE; i++) {
        b[i] = 0.0;
    }

    #pragma acc parallel loop present(b)
    for (int j = 0; j < NY; j++) {
        double t = static_cast<double>(j) / (NY - 1);
        b[j * NX] = corner_values[0] * (1 - t) + corner_values[3] * t;
        b[j * NX + (NX - 1)] = corner_values[1] * (1 - t) + corner_values[2] * t;
    }

    #pragma acc parallel loop present(b)
    for (int i = 0; i < NX; i++) {
        double t = static_cast<double>(i) / (NX - 1);
        b[i] = corner_values[0] * (1 - t) + corner_values[1] * t;
        b[(NY - 1) * NX + i] = corner_values[3] * (1 - t) + corner_values[2] * t;
    }
}

void init_matrix(double *A) {
    #pragma acc enter data create(A[0:SIZE*SIZE])
    
    #pragma acc parallel loop present(A)
    for (int i = 0; i < SIZE; i++) {
        #pragma acc loop
        for (int j = 0; j < SIZE; j++) {
            A[i*SIZE + j] = get_a(i, j); 
        }
    }
}

// ЗАМЕНА cublasDnrm2
double norm(const double *x) {
    double sum = 0.0;
    #pragma acc parallel loop reduction(+:sum) present(x)
    for (int i = 0; i < SIZE; i++) {
        sum += x[i] * x[i];
    }
    return sqrt(sum);
}

void mul_mv_sub(double *res, const double *A, const double *x, const double *y) {
    #pragma acc parallel loop present(res, A, x, y)
    for (int i = 0; i < SIZE; ++i) {
        double sum = -y[i];
        #pragma acc loop seq
        for (int j = 0; j < SIZE; ++j) {
            sum += A[i * SIZE + j] * x[j];
        }
        res[i] = sum;
    }
}

void next(double *x, const double *delta) {
    #pragma acc parallel loop present(x, delta)
    for (int i = 0; i < SIZE; i++) {
        x[i] -= TAU * delta[i];
    }
}

void solve_simple_iter(const double *A, double *x, const double *b) {
    double* Axmb = (double*)malloc(SIZE * sizeof(double));
    #pragma acc enter data create(Axmb[0:SIZE])
    
    double norm_b = norm(b);
    double norm_Axmb;
    int iter = 0;

    do {
        mul_mv_sub(Axmb, A, x, b);
        norm_Axmb = norm(Axmb);
        next(x, Axmb);
        
        std::cout << "Iteration " << ++iter 
                  << ": residual = " << norm_Axmb/norm_b << " (target < " << EPS << ")\r";
        std::cout.flush();
        
        if (iter >= MAX_ITER) {
            std::cout << "\nMaximum iterations (" << MAX_ITER << ") reached\n";
            break;
        }
    } while (norm_Axmb/norm_b >= EPS);

    if (iter < MAX_ITER) {
        std::cout << "\nConverged after " << iter << " iterations\n";
    }
    
    #pragma acc exit data delete(Axmb)
    free(Axmb);
}

void save_results(const double* x, int size) {
    #pragma acc update self(x[0:size])
    std::ofstream out(OUT_FILE, std::ios::binary);
    out.write(reinterpret_cast<const char*>(x), size * sizeof(double));
}

// Новая функция для сохранения результата в текстовом формате
void save_text_output(const double* x, int nx, int ny, const std::string& filename) {
    #pragma acc update self(x[0:nx * ny])
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Failed to open " << filename << " for writing\n";
        return;
    }

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            out << x[j * nx + i] << " ";
        }
        out << "\n";
    }

    out.close();
}

int main() {
    double* A = (double*)malloc(SIZE * SIZE * sizeof(double));
    double* b = (double*)malloc(SIZE * sizeof(double));
    double* x = (double*)malloc(SIZE * sizeof(double));

    #pragma acc enter data create(x[0:SIZE]) copyin(corner_values[0:4])
    
    #pragma acc parallel loop present(x)
    for (int i = 0; i < SIZE; ++i) {
        x[i] = 0.0;
    }

    std::cout << "Solving heat distribution on " << NY << "x" << NX << " grid\n";
    std::cout << "Parameters: tau=" << TAU << ", eps=" << EPS << ", max_iter=" << MAX_ITER << "\n";

    init_matrix(A);
    init_b(b);

    auto start = high_resolution_clock::now();

    solve_simple_iter(A, x, b);

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    printf("Time run = %.5f s\n",float(duration.count())/1000000);

    save_results(x, SIZE);
    save_text_output(x, NX, NY, "output2.txt");

    #pragma acc exit data delete(A, b, x)
    free(A);
    free(b);
    free(x);

    return 0;
}
