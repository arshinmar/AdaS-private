import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
def curve_fit(x_data,y_data):
    def func(x, a, b, c):
        return a * np.exp(-b * x) + c
    popt, pcov = curve_fit(func, x_data, y_data)
    plt.plot(xdata, func(xdata, *popt), 'r-',
             label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    plt.show()

epoch_number=[]
