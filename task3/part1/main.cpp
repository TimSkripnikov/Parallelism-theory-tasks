#include <iostream>
#include <vector>
#include <thread>
#include <fstream>
#include <chrono>
#include <functional>

void initialize_matrix(std::vector<double> &matrix, int start_idx, int end_idx, int n)
{
    for (int i = start_idx; i < end_idx; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            matrix[i * n + j] = j + j;
        }
    }
}

void initialize_vector(std::vector<double> &vector, int start_idx, int end_idx)
{
    for (int j = start_idx; j < end_idx; ++j)
    {
        vector[j] = j;
    }
}

void matrix_vector_multiplication(std::vector<double> &matrix, std::vector<double> &vector, std::vector<double> &result, int start_idx, int end_idx, int n)
{
    for (int i = start_idx; i < end_idx; ++i) {
        result[i] = 0.0;
        for (int j = 0; j < n; ++j)
            result[i] += matrix[i * n + j] * vector[j];
    }
}

double run_threaded(int n, int num_threads) {
    std::vector<double> matrix(n * n);
    std::vector<double> vector(n);
    std::vector<double> result(n);

    {
        std::vector<std::thread> threads;
        int block = n / num_threads;

        for (int t = 0; t < num_threads; ++t) {
            int start_idx = t * block;
            int end_idx = (t == num_threads - 1) ? n : start_idx + block;
            threads.emplace_back(initialize_matrix, std::ref(matrix), start_idx, end_idx, n);
        }
        for (auto &t : threads) t.join();
    }

    {
        std::vector<std::thread> threads;
        int block = n / num_threads;

        for (int t = 0; t < num_threads; ++t) {
            int start_idx = t * block;
            int end_idx = (t == num_threads - 1) ? n : start_idx + block;
            threads.emplace_back(initialize_vector, std::ref(vector), start_idx, end_idx);
        }
        for (auto &t : threads) t.join();
    }

    auto start = std::chrono::high_resolution_clock::now();

    {
        std::vector<std::thread> threads;
        int block = n / num_threads;

        for (int t = 0; t < num_threads; ++t) {
            int start_idx = t * block;
            int end_idx = (t == num_threads - 1) ? n : start_idx + block;
            threads.emplace_back(matrix_vector_multiplication,
                                 std::ref(matrix), std::ref(vector), std::ref(result),
                                 start_idx, end_idx, n);
        }
        for (auto &t : threads) t.join();
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    return diff.count();
}

void writeCSV(const char *filename, double results[2][16]) {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Error writing file\n";
        exit(1);
    }

    file << "N=M,T1,T2,S2,T4,S4,T7,S7,T8,S8,T16,S16,T20,S20,T40,S40\n";

    for (int i = 0; i < 2; ++i) {
        file << ((i == 0) ? 20000 : 40000);
        for (int j = 0; j < 15; ++j) {
            file << "," << results[i][j];
        }
        file << "\n";
    }
    file.close();
}

int main() {
    int thread_counts[] = {2, 4, 7, 8, 16, 20, 40};
    double results[2][16] = {0};
    const char *filename = "results_thread.csv";

    for (int test = 0; test < 2; ++test) {
        int size = (test == 0) ? 20000 : 40000;
        double time_serial = run_threaded(size, 1);
        results[test][0] = time_serial;

        for (int i = 0; i < 7; ++i) {
            int threads = thread_counts[i];
            double time_parallel = run_threaded(size, threads);
            results[test][2 * i + 1] = time_parallel;
            results[test][2 * i + 2] = time_serial / time_parallel;
        }
    }

    writeCSV(filename, results);
    std::cout << "Done. Results written to " << filename << std::endl;
    return 0;
}
