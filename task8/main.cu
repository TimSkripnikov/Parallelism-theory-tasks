#include "cuda_runtime.h"
#include <stdio.h>

__global__ void HelloWorld()
{
    printf("hello, %d, %d\n", blockIdx.x, threadIdx.x);


}

int main()
{
    HelloWorld <<<2, 5>>>();

    cudaDeviceSynchronize();
    return 0;
}