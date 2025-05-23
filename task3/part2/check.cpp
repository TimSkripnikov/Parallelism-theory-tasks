#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <string>
#include <iomanip>

bool is_equal(double a, double b, double eps = 1e-6) {
    if (std::fabs(a - b) < eps) return true;
    double rel_err = std::fabs(a - b) / std::max(std::fabs(a), std::fabs(b));
    return rel_err < 1e-4;
}

void check_file(const std::string& filename) {
    std::ifstream fin(filename);
    if (!fin.is_open()) {
        std::cerr << "Failed to open file: " << filename << "\n";
        return;
    }

    std::string line;
    int line_num = 0;
    int errors = 0;

    while (std::getline(fin, line)) {
        ++line_num;

        std::istringstream iss(line);
        std::string op;
        
        double arg1, arg2 = 0.0, expected;

        iss >> op >> arg1;
        if (op == "pow") {
            iss >> arg2;
        }
        std::string eq;
        iss >> eq >> expected;

        double computed = 0.0;
        if (op == "sin") {
            computed = std::sin(arg1);
        } else if (op == "sqrt") {
            computed = std::sqrt(arg1);
        } else if (op == "pow") {
            computed = std::pow(arg1, arg2);
        } else {
            std::cerr << "Unknown operation at line " << line_num << ": " << op << "\n";
            continue;
        }

        if (!is_equal(expected, computed)) {
            std::cout << "Mismatch at line " << line_num << " → "
                      << op << " " << arg1;
            if (op == "pow") std::cout << " " << arg2;
            std::cout << ": expected " << expected << ", got " << computed << "\n";
            ++errors;
        }
    }

    if (errors == 0) {
        std::cout << "File " << filename << ": all values are correct \n";
    } else {
        std::cout << "File " << filename << ": total mismatches: " << errors << "\n";
    }
}

int main() {
    check_file("sin_output.txt");
    check_file("sqrt_output.txt");
    check_file("pow_output.txt");

    std::cout << "All checks completed.\n";
    return 0;
}
