import pycuda.autoinit
import re, sys
import numpy as np
import numba
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import pycuda.driver as cuda
from multiprocessing import Pool
from functools import partial
from pycuda.gpuarray import GPUArray
from pycuda.compiler import SourceModule


mod = SourceModule("""
#include <math.h>

__device__ int phase(float x0, float y0, float x1, float y1)
{    
    bool realGreater = (x0 >= x1);
    bool imagGreater = (y0 >= y1);

    if (realGreater && imagGreater)
    {
        return 1;
    }
    else if (!realGreater && imagGreater)
    {
        return 2;
    }
    else if (!realGreater && !imagGreater)
    {
        return 3;
    };

    return 4;
}

__device__ float calculateNewCentreX(float x1, int previousResult, int n)
{
    if (previousResult == 1 || previousResult == 4)
    {
        return x1 + powf(2, n - 2);
    };

    return x1 - powf(2, n - 2);
}

__device__ float calculateNewCentreY(float y1, int previousResult, int n)
{
    if (previousResult == 1 || previousResult == 2)
    {
        return y1 + powf(2, n - 2);
    };

    return y1 - powf(2, n - 2);
}

__global__ void getGridValue(int *regexStringGrid, int n)
{
    int real = blockDim.x * blockIdx.x + threadIdx.x;
    int imag = blockDim.y * blockIdx.y + threadIdx.y;
    int dim = gridDim.x * (blockIdx.y * blockDim.y + threadIdx.y) + blockDim.x * blockIdx.x + threadIdx.x;

    float x1 = powf(2, n - 1);
    float y1 = powf(2, n - 1);

    for (int i = 0; i < n; ++i)
    {
        int value = phase(real, imag, x1, y1);
        x1 = calculateNewCentreX(x1, value, n - i);
        y1 = calculateNewCentreY(y1, value, n - i);
        regexStringGrid[dim] += value * powf(10, i);
    }
}""")

def get_grid_string(n, i, j):
    result =  ''
    currentCentre = complex(2**(n-1), 2**(n-1))
    for k in range(n):
        if i >= currentCentre.real and j >= currentCentre.imag:
            result += '1'
            currentCentre += complex(2 ** (n - 2 - k)) * complex(1, 1)
        elif i < currentCentre.real and j >= currentCentre.imag:
            result += '2'
            currentCentre += complex(2 ** (n - 2 - k)) * complex(-1, 1)
        elif i >= currentCentre.real and j < currentCentre.imag:
            result += '3'
            currentCentre += complex(2 ** (n - 2 - k)) * complex(1, -1)
        else:
            result += '4'
            currentCentre += complex(2 ** (n - 2 - k)) * complex(-1, -1)

    return result

def regex_fractal(n, regex=None):
    new_grid = np.zeros((2**n, 2**n))
    for i in range(new_grid.shape[0]):
        for j in range(new_grid.shape[1]):
            y = get_grid_string(i, j, n)
            x = re.match(regex, y)
            if x:
                group_lengths = sum([len(y) for y in x.groups()])
                new_grid[i,j] =  group_lengths * 50
            
    return new_grid

if __name__ == "__main__":
    n = 5
    x = np.zeros((2**n, 2**n, 1))
    get_grid_value = mod.get_function('getGridValue')
    block = (4, 4, 1)
    grid = (2**n // block[0], 2**n // block[1], 1)

    start = timer()
    regex = '.*(?:13|21)(.*)'
    get_grid_value(cuda.Out(x), cuda.In(np.array([n])), block=block, grid=grid)
    end = timer()
    print("Time taken {}".format(end - start))
    print(x)


