#include <iostream>
#include <vector>
#include <cmath>

#ifndef USE_DOUBLE
typedef float invariant_t;
#else
typedef double invariant_t;
#endif

const size_t n = 10000000;

int main()
{
    std::vector<invariant_t> data(n);

    for (size_t i = 0; i < n; ++i)
    {
        data[i] = std::sin((2 * M_PI * i) / n);
    }

    invariant_t sum = 0;
    for (size_t i = 0; i < n; ++i)
    {
        sum += data[i];
    }

    std::cout << "Sum: " << sum << std::endl;
}
