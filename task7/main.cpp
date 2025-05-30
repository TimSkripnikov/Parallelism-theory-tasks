#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <cassert>
#include <openacc.h>

#include "cublas_v2.h"
#include <boost/program_options.hpp>

namespace bpo = boost::program_options;

double lin_interpolation(double x, double x1, double y1, double x2, double y2) {
    return y1 + ((x - x1) * (y2 - y1) / (x2 - x1));
}

void init(std::unique_ptr<double[]> &A, int size) {
    A[0] = 10.0;
    A[size - 1] = 20.0;
    A[(size - 1) * size + (size - 1)] = 30.0;
    A[(size - 1) * size] = 20.0;

    for (size_t i = 1; i < size - 1; ++i) {
        A[i] = lin_interpolation(i, 0.0, A[0], size - 1, A[size - 1]);
        A[i * size] = lin_interpolation(i, 0.0, A[0], size - 1, A[(size - 1) * size]);
        A[i * size + (size - 1)] = lin_interpolation(i, 0.0, A[size - 1], size - 1, A[(size - 1) * size + (size - 1)]);
        A[(size - 1) * size + i] = lin_interpolation(i, 0.0, A[(size - 1) * size], size - 1, A[(size - 1) * size + (size - 1)]);
    }
}

int save_to_file(const double* A, int size, const std::string& filename) {
    std::ofstream f(filename);
    if (!f.is_open()) return 1;

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            f << A[i * size + j] << " ";
        }
        f << "\n";
    }

    f.close();
    return 0;
}

int main(int argc, const char** argv) {
    bpo::options_description desc("Options");
    desc.add_options()
        ("help", "Show help message")
        ("size", bpo::value<int>()->default_value(128), "Grid size")
        ("num_iters", bpo::value<int>()->default_value(1000000), "Max number of iterations")
        ("eps", bpo::value<double>()->default_value(1e-6), "Desired precision");

    bpo::variables_map vm;
    bpo::store(bpo::parse_command_line(argc, argv, desc), vm);
    bpo::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }

    int size = vm["size"].as<int>();
    int num_iters = vm["num_iters"].as<int>();
    double eps = vm["eps"].as<double>();

    acc_set_device_num(2, acc_device_nvidia);

    cublasStatus_t status;
    cublasHandle_t cublasHandle;
    status = cublasCreate(&cublasHandle);

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "CUBLAS initialization failed\n";
        return 3;
    }

    double error = 1.0;
    int iters = 0;

    std::unique_ptr<double[]> A(new double[size * size]);
    std::unique_ptr<double[]> new_A(new double[size * size]);

    double alpha = -1.0;
    int max_idx = 0;

    init(A, size);
    init(new_A, size);

    auto start = std::chrono::high_resolution_clock::now();

    double* first_matrix = A.get();
    double* second_matrix = new_A.get();

    #pragma acc data copyin(max_idx, size, alpha, second_matrix[0:size * size], first_matrix[0:size * size])
    {
        while (iters < num_iters && error > eps) {
            #pragma acc parallel loop independent collapse(2) vector vector_length(1024) gang num_gangs(1024) present(first_matrix, second_matrix)
            for (size_t i = 1; i < size - 1; ++i) {
                for (size_t j = 1; j < size - 1; ++j) {
                    first_matrix[i * size + j] = 0.25 * (
                        second_matrix[i * size + j + 1] +
                        second_matrix[i * size + j - 1] +
                        second_matrix[(i - 1) * size + j] +
                        second_matrix[(i + 1) * size + j]
                    );
                }
            }

            if ((iters + 1) % 10000 == 0) {
                #pragma acc data present(second_matrix, first_matrix) wait
                #pragma acc host_data use_device(first_matrix, second_matrix)
                {
                    status = cublasDaxpy(cublasHandle, size * size, &alpha, first_matrix, 1, second_matrix, 1);
                    if (status != CUBLAS_STATUS_SUCCESS) {
                        std::cerr << "CUBLAS_Daxpy failed\n";
                    }

                    status = cublasIdamax(cublasHandle, size * size, second_matrix, 1, &max_idx);
                    if (status != CUBLAS_STATUS_SUCCESS) {
                        std::cerr << "CUBLAS_ID_max failed\n";
                    }
                }

                #pragma acc update self(second_matrix[max_idx - 1])
                error = fabs(second_matrix[max_idx - 1]);

                #pragma acc host_data use_device(first_matrix, second_matrix)
                {
                    status = cublasDcopy(cublasHandle, size * size, first_matrix, 1, second_matrix, 1);
                    if (status != CUBLAS_STATUS_SUCCESS) {
                        std::cerr << "CUBLAS_Dcopy failed\n";
                    }
                }

                
            }

            double* tmp = first_matrix;
            first_matrix = second_matrix;
            second_matrix = tmp;

            ++iters;
        }

        cublasDestroy(cublasHandle);
        #pragma acc update self(first_matrix[0:size * size])
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

    std::cout << "Elapsed time (msec): " << elapsed << std::endl;
    std::cout << "Iterations: " << iters  << ", Error: " << error << std::endl;

    if (size == 13 || size == 10) {
        for (size_t i = 0; i < size; ++i) {
            for (size_t j = 0; j < size; ++j) {
                std::cout << A[i * size + j] << ' ';
            }
            std::cout << std::endl;
        }
    }

    save_to_file(first_matrix, size, "output.txt");

    A = nullptr;
    new_A = nullptr;

    return 0;
}

//  nsys profile --output=full_report1 ./heat_gpu --size 512 --num_iters 100000
