from PIL import Image
import re, sys
import numpy as np
from timeit import default_timer as timer

class complex_number:
    """ complex number class """
    def __init__(self, real, imag):
        self.real = real
        self.imag = imag

    def __repr__(self):
        return (str(self.real) + "+" + str(self.imag) + "i")

    # Complex addition
    def cadd(self, other):
        return complex_number(self.real + other.real, self.imag + other.imag)

    # Complex multiplication
    def cmultiply(self, other):
        return complex_number(self.real * other.real - self.imag * other.imag, self.real * other.imag + self.imag * other.real)
    
    # Complex absolute value
    def norm(self):
        return sqrt(self.real ** 2 + self.imag ** 2)
    
    # Complex conjugate
    def conjugate(self):
        return complex_number(self.real, - self.imag)
    
    # Complex division
    def cdivide(self, other):
        numerator = self.cmultiply(other.conjugate())
        denominator = float(other.norm() ** 2)
        return complex(numerator.real / denominator, numerator.imag / denominator)

    # Complex argument
    def argument(self):

        x = self.real * 1.0
        y = self.imag * 1.0

        if x > 0:
            return np.arctan(y / x)
        elif x < 0:
            return np.arctan(y / x) + np.sign(np.sign(y) + 0.5) * np.pi
        elif x == 0 and y != 0:
            return np.sign(y) * np.pi / 2
        else:
            return "undefined"

def grid_numbering(n, x_0, y_0, x_1, y_1):
    """generates the grid number for the regex grid"""
   
    if n == 0:
        return ""

    arg = complex_number(x_0 + 0.5 - x_1, y_0 + 0.5 - y_1).argument()

    if arg >= 0 and arg < np.pi / 2: 
        x = "1"
        x_1 += 2 ** (n - 2)
        y_1 += 2 ** (n - 2)
    elif arg >= np.pi / 2 and arg <= np.pi:
        x = "2"
        x_1 -= 2 ** (n - 2)
        y_1 += 2 ** (n - 2)
    elif arg < 0 and arg >= -np.pi / 2:
        x = "4"
        x_1 += 2 ** (n - 2)
        y_1 -= 2 ** (n - 2)
    else:
        x = "3"
        x_1 -= 2 ** (n - 2)
        y_1 -= 2 ** (n - 2)

    return str(x) + grid_numbering(n - 1, x_0, y_0, x_1, y_1)

def regex_grid(n):
    """create the regex grid that contains every number"""
    cx = 2 ** (n - 1)
    cy = 2 ** (n - 1)
    grid = [[grid_numbering(n, i , j, cx, cy) for i in range(2 ** n)] for j in range(2 ** n)]
    
    return grid

def regex_fractal(n, regex):
    start = timer()
    grid = regex_grid(n)
    print("Time taken to create regex grid {}".format(timer() - start))

    img = Image.new( 'RGB', (len(grid), len(grid)), "white")
    pixels = img.load()
    regex = '.*(?:13|321)(.*)'
    # regex = raw_input("Regex please (example .*(?:13|321)(.*) .*1(.*):")
    #regex = '.*1(.*)'

    start = timer()
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            x = re.match(regex, grid[i][j])
            if x:
                group_lengths = sum(len(y) for y in x.groups())
                pixels[i, j] =  (group_lengths * 50, 0, 0)
    print("Time taken to color the pixels {}".format(timer() - start))

    img.show()

if __name__ == "__main__":
    user_input = input("Please give me a regular expression: ")
    regex_fractal(10, user_input)

            

