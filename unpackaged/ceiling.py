import math
import numpy as np

def nearest_upper_odd(squared_kernel_size):
    actual_kernel_size = np.sqrt(np.array(squared_kernel_size))#.float())
    return np.ceil(actual_kernel_size) // 2 * 2 + 1

squared_kernel_size=[[1.1,1.1],[1.1,1.1]]
print(nearest_upper_odd(squared_kernel_size))
