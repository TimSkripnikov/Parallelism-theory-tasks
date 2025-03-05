#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>

int main()
{
    int n = 30;
    double eps = 0.000001;
    double t = 0.01;

    std::vector<std::vector<double>> A(n, std::vector<double>(n, 1.0));
    std::vector<double> B(n, n + 1);
    std::vector<double> X(n, 0.0);
    std::vector<double> X_new(n, 0.0);

    // Заполняем матрицу A
    for (int i = 0; i < n; ++i)
    {
        A[i][i] = 2.0;
    }

    // Вычисляем норму B (знаменатель)
    double norm_B = 0.0;
#pragma omp parallel for reduction(+ : norm_B)
    for (int i = 0; i < n; ++i)
    {
        norm_B += B[i] * B[i];
    }
    norm_B = std::sqrt(norm_B);

    int iter = 0; // Счетчик итераций

    // Итерационный процесс
    while (true)
    {
        ++iter;

// Параллелизация вычисления новой итерации X_new
#pragma omp parallel for
        for (int i = 0; i < n; ++i)
        {
            double S = 0.0;
// Параллелизация суммы по j
#pragma omp parallel for reduction(+ : S)
            for (int j = 0; j < n; ++j)
            {
                S += A[i][j] * X[j];
            }
            X_new[i] = X[i] - t * (S - B[i]);
        }

        // Параллелизация вычисления нормы невязки
        double norm_residual = 0.0;
#pragma omp parallel for reduction(+ : norm_residual)
        for (int i = 0; i < n; ++i)
        {
            double S = 0.0;
// Параллелизация суммы по j
#pragma omp parallel for reduction(+ : S)
            for (int j = 0; j < n; ++j)
            {
                S += A[i][j] * X_new[j];
            }
            norm_residual += (S - B[i]) * (S - B[i]);
        }
        norm_residual = std::sqrt(norm_residual);

        // Проверяем критерий остановки
        if (norm_residual / norm_B < eps)
        {
            break;
        }

// Обновляем X
#pragma omp parallel for
        for (int i = 0; i < n; ++i)
        {
            X[i] = X_new[i];
        }
    }

    // Вывод результата
    std::cout << "Решение найдено за " << iter << " итераций." << std::endl;
    std::cout << "Вектор решений X:" << std::endl;
    for (int i = 0; i < n; ++i)
    {
        std::cout << X[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
