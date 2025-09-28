import numpy as np

# Introduction to NumPy: A Beginner's Guide

# What is NumPy?
# NumPy (Numerical Python) is a powerful library for numerical computing in Python.
# It provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on them.

# Why use NumPy?
# - Efficient storage and manipulation of numerical data
# - Fast mathematical operations
# - Useful for data science, machine learning, scientific computing, and more

# Let's get started!

# 1. Importing NumPy

# 2. Creating Arrays

# The basic building block in NumPy is the array.
# You can create an array from a Python list using np.array()

a = np.array([1, 2, 3, 4, 5])
print("Array a:", a)

# Arrays can be multi-dimensional (e.g., 2D arrays, also called matrices)
b = np.array([[1, 2, 3], [4, 5, 6]])
print("2D Array b:\n", b)

# 3. Array Properties

print("Shape of a:", a.shape)  # Number of elements in each dimension
print("Shape of b:", b.shape)
print("Data type of a:", a.dtype)  # Type of elements (int, float, etc.)
print("Number of dimensions in b:", b.ndim)

# 4. Creating Arrays with Zeros, Ones, and Other Values

zeros = np.zeros((2, 3))  # 2 rows, 3 columns, filled with zeros
print("Zeros array:\n", zeros)

ones = np.ones((3, 2))  # 3 rows, 2 columns, filled with ones
print("Ones array:\n", ones)

full = np.full((2, 2), 7)  # 2x2 array filled with 7
print("Full array:\n", full)

# 5. Creating Arrays with Ranges

# np.arange() creates arrays with regularly spaced values
arr = np.arange(0, 10, 2)  # Start at 0, end before 10, step by 2
print("Array with range:", arr)

# np.linspace() creates arrays with evenly spaced values between two numbers
arr2 = np.linspace(0, 1, 5)  # 5 values from 0 to 1
print("Linspace array:", arr2)

# 6. Random Arrays

rand_arr = np.random.rand(2, 3)  # 2x3 array with random floats between 0 and 1
print("Random array:\n", rand_arr)

rand_int = np.random.randint(0, 10, (3, 3))  # 3x3 array with random integers from 0 to 9
print("Random integer array:\n", rand_int)

# 7. Indexing and Slicing

# Access elements by index (starts at 0)
print("First element of a:", a[0])
print("First row of b:", b[0])
print("Element at row 1, column 2 of b:", b[1, 2])

# Slicing: get a subset of an array
print("First three elements of a:", a[:3])
print("First two columns of b:\n", b[:, :2])

# 8. Array Operations

# Arithmetic operations are element-wise
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

print("x + y:", x + y)
print("x * y:", x * y)
print("x squared:", x ** 2)

# Mathematical functions
print("Sum of x:", np.sum(x))
print("Mean of y:", np.mean(y))
print("Maximum of y:", np.max(y))

# 9. Reshaping Arrays

# Change the shape of an array with reshape()
c = np.arange(12)
print("Original c:", c)
c_reshaped = c.reshape((3, 4))  # 3 rows, 4 columns
print("Reshaped c:\n", c_reshaped)

# 10. Useful Array Methods

# Transpose: swap rows and columns
print("Transpose of b:\n", b.T)

# Flatten: convert multi-dimensional array to 1D
print("Flattened b:", b.flatten())

# 11. Broadcasting

# NumPy can automatically expand arrays to match shapes for operations
d = np.array([1, 2, 3])
e = np.array([[10], [20], [30]])
print("d + e:\n", d + e)  # e is "broadcast" to match d's shape

# 12. Saving and Loading Arrays

# Save to file
np.save('my_array.npy', a)

# Load from file
loaded_a = np.load('my_array.npy')
print("Loaded array:", loaded_a)

# 13. Summary

# NumPy is a powerful tool for numerical computing.
# Key concepts:
# - Arrays (1D, 2D, etc.)
# - Array creation (from lists, ranges, random values)
# - Indexing, slicing, and reshaping
# - Mathematical operations and functions
# - Broadcasting
# - Saving/loading arrays

# For more information, check out the official documentation:
# https://numpy.org/doc/

# Practice: Try creating your own arrays and performing operations!