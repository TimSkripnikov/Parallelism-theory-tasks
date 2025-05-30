// main.cu

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <fstream>
#include <memory>
#include <cassert>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <boost/program_options.hpp>

namespace bpo = boost::program_options;

__global__ void jacobi_kernel(double* out, const double* in, int size) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i > 0 && i < size - 1 && j > 0 && j < size - 1) {
        int idx = i * size + j;
        out[idx] = 0.25 * (
            in[idx + 1] + in[idx - 1] +
            in[idx + size] + in[idx - size]
        );
    }
}

__global__ void compute_diff_kernel(double* diff, const double* A, const double* B, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        diff[idx] = fabs(A[idx] - B[idx]);
    }
}

double lin_interp(double x, double x1, double y1, double x2, double y2) {
    return y1 + ((x - x1) * (y2 - y1)) / (x2 - x1);
}

void init(std::vector<double>& A, int size) {
    A[0] = 10.0;
    A[size - 1] = 20.0;
    A[(size - 1) * size + (size - 1)] = 30.0;
    A[(size - 1) * size] = 20.0;

    for (int i = 1; i < size - 1; ++i) {
        A[i] = lin_interp(i, 0.0, A[0], size - 1, A[size - 1]);
        A[i * size] = lin_interp(i, 0.0, A[0], size - 1, A[(size - 1) * size]);
        A[i * size + (size - 1)] = lin_interp(i, 0.0, A[size - 1], size - 1, A[(size - 1) * size + (size - 1)]);
        A[(size - 1) * size + i] = lin_interp(i, 0.0, A[(size - 1) * size], size - 1, A[(size - 1) * size + (size - 1)]);
    }
}

double compute_max_error(double* A, double* B, int N) {
    double* d_diff;

    cudaMalloc(&d_diff, N * sizeof(double));

    int threads = 256;

    int blocks = (N + threads - 1) / threads;

    compute_diff_kernel<<<blocks, threads>>>(d_diff, A, B, N);


    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;


    double* d_result;
    cudaMalloc(&d_result, sizeof(double));

    cub::DeviceReduce::Max(
        d_temp_storage, 
        temp_storage_bytes, 
        d_diff, 
        d_result, 
        N
    );

    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    
    cub::DeviceReduce::Max(
        d_temp_storage, 
        temp_storage_bytes, 
        d_diff, 
        d_result, 
        N
    );

    double h_result;

    cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_diff);
    cudaFree(d_temp_storage);
    cudaFree(d_result);
    return h_result;
}

void save_to_file(const std::vector<double>& A, int size, const std::string& filename) {
    std::ofstream f(filename);
    if (!f.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            f << A[i * size + j] << " ";
        }
        f << "\n";
    }
    f.close();
}

int main(int argc, char** argv) {
    int size, max_iters;
    double eps;

    bpo::options_description desc("Options");
    desc.add_options()
        ("help,h", "Show help")
        ("size", bpo::value<int>(&size)->default_value(128), "Grid size")
        ("num_iters", bpo::value<int>(&max_iters)->default_value(1000000), "Max iterations")
        ("eps", bpo::value<double>(&eps)->default_value(1e-6), "Precision");

    bpo::variables_map vm;
    bpo::store(bpo::parse_command_line(argc, argv, desc), vm);
    bpo::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }

    int N = size * size;

    std::vector<double> host_A(N, 0.0), host_B(N, 0.0);
    init(host_A, size);
    init(host_B, size);

    double *d_A, *d_B;

    cudaMalloc(&d_A, N * sizeof(double));
    cudaMalloc(&d_B, N * sizeof(double));

    cudaMemcpy(d_A, host_A.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, host_B.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((size + block.x - 1) / block.x, (size + block.y - 1) / block.y);

    int iters = 0;
    int k = 10000;
    double error = 1.0;

    cudaStream_t stream;
    cudaStreamCreate(&stream);   //

    cudaGraph_t graph;
    cudaGraphExec_t instance;

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    for (int i = 0; i < k; ++i) {
        jacobi_kernel<<<grid, block, 0, stream>>>(d_B, d_A, size);
        std::swap(d_A, d_B);
    }

    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);

    auto start = std::chrono::high_resolution_clock::now();

    while (iters < max_iters && error > eps) {
        cudaGraphLaunch(instance, stream);
        cudaStreamSynchronize(stream);  //Ожидает завершения всех задач в stream, то есть ждет, пока GPU закончит вычисления, запущенные графом

        error = compute_max_error(d_A, d_B, N);
        std::swap(d_A, d_B);
        iters += k;
    }
    cudaMemcpy(host_A.data(), d_A, N * sizeof(double), cudaMemcpyDeviceToHost);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time (ms): " << elapsed.count() * 1000 << std::endl;
    std::cout << "Iterations: " << iters << ", Error: " << error << std::endl;

    save_to_file(host_A, size, "output.txt");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaStreamDestroy(stream);
    cudaGraphDestroy(graph);
    cudaGraphExecDestroy(instance);

    return 0;
}
