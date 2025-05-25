#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <cassert>
#include <openacc.h>
#include <fstream>
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
        ("help,h", "Show help message")
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

    acc_set_device_num(3, acc_device_nvidia);

    double error = 1.0;
    int iters = 0;

    std::unique_ptr<double[]> A(new double[size * size]);
    std::unique_ptr<double[]> new_A(new double[size * size]);

    init(A, size);
    init(new_A, size);

    
    auto start = std::chrono::high_resolution_clock::now();

    double* currentMatrix = A.get();
    double* previousMatrix = new_A.get();

    #pragma acc data copyin(error, previousMatrix[0:size * size], currentMatrix[0:size * size])
    {
        while (iters < num_iters && error > eps) {
            #pragma acc parallel loop independent collapse(2) present(currentMatrix, previousMatrix)

            for (size_t i = 1; i < size - 1; ++i) {
                for (size_t j = 1; j < size - 1; ++j) {
                    currentMatrix[i * size + j] = 0.25 * (
                        previousMatrix[i * size + j + 1] + 
                        previousMatrix[i * size + j - 1] + 
                        previousMatrix[(i - 1) * size + j] + 
                        previousMatrix[(i + 1) * size + j]
                    );
                }
            }

            if ((iters + 1) % 10000 == 0) {
                error = 0.0;
                #pragma acc update device(error)
                
                #pragma acc parallel loop independent collapse(2) reduction(max:error) present(currentMatrix, previousMatrix)
                for (size_t i = 1; i < size - 1; ++i) {
                    for (size_t j = 1; j < size - 1; ++j) {
                        error = fmax(error, fabs(currentMatrix[i * size + j] - previousMatrix[i * size + j]));
                    }
                }

                #pragma acc update self(error)
            }

            double* temp = currentMatrix;
            currentMatrix = previousMatrix;
            previousMatrix = temp;
    
            ++iters;

            }
        #pragma acc update self(currentMatrix[0:size * size])
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

    std::cout << "Elapsed time (msec): " << elapsed << std::endl;
    std::cout << "Iterations: " << iters + 1 << ", Error: " << error << std::endl;

    if (size == 13 || size == 10) {
        for (size_t i = 0; i < size; ++i) {
            for (size_t j = 0; j < size; ++j) {
                std::cout << A[i * size + j] << ' ';
            }
            std::cout << std::endl;
        }
    }

    save_to_file(currentMatrix, size, "output.txt");
    
    A = nullptr;
    new_A = nullptr;

    return 0;
}