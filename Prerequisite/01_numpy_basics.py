"""
NumPy Fundamentals for Data Science
====================================
NumPy is the foundation of numerical computing in Python.
It provides fast array operations and mathematical functions.

Author: ML Teacher (5 years experience)
Date: 2026
"""

import numpy as np

# ============================================================================
# 1. CREATING ARRAYS
# ============================================================================

print("\n" + "="*70)
print("1. CREATING NUMPY ARRAYS")
print("="*70)

# From Python lists
array_1d = np.array([1, 2, 3, 4, 5])
print(f"\n1D Array: {array_1d}")
print(f"Shape: {array_1d.shape}, Data Type: {array_1d.dtype}")

# 2D array
array_2d = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])
print(f"\n2D Array:\n{array_2d}")
print(f"Shape: {array_2d.shape}")

# Common creation methods
zeros = np.zeros((3, 3))  # Array of zeros
ones = np.ones((2, 4))    # Array of ones
identity = np.eye(3)       # Identity matrix
range_array = np.arange(0, 10, 2)  # Range: start, stop, step
linspace = np.linspace(0, 1, 5)   # 5 evenly spaced values from 0-1

print(f"\nZeros:\n{zeros}")
print(f"\nIdentity Matrix:\n{identity}")
print(f"\nRange Array: {range_array}")
print(f"Linspace: {linspace}")


# ============================================================================
# 2. ARRAY OPERATIONS
# ============================================================================

print("\n" + "="*70)
print("2. ARRAY OPERATIONS")
print("="*70)

a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

print(f"\nArray A: {a}")
print(f"Array B: {b}")

# Element-wise operations
print(f"\nAddition (A + B): {a + b}")
print(f"Subtraction (A - B): {a - b}")
print(f"Multiplication (A * B): {a * b}")
print(f"Division (A / B): {a / b}")
print(f"Power (A ** 2): {a ** 2}")

# Aggregation operations
print(f"\nSum of A: {np.sum(a)}")
print(f"Mean of A: {np.mean(a)}")
print(f"Std Dev of A: {np.std(a)}")
print(f"Min of A: {np.min(a)}, Max of A: {np.max(a)}")


# ============================================================================
# 3. INDEXING AND SLICING
# ============================================================================

print("\n" + "="*70)
print("3. INDEXING AND SLICING")
print("="*70)

array = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
matrix = np.arange(1, 10).reshape(3, 3)

print(f"\n1D Array: {array}")
print(f"First element: {array[0]}")
print(f"Last element: {array[-1]}")
print(f"Slice [1:5]: {array[1:5]}")
print(f"Every 2nd element: {array[::2]}")

print(f"\nMatrix:\n{matrix}")
print(f"Element at [1, 2]: {matrix[1, 2]}")
print(f"First row: {matrix[0, :]}")
print(f"First column: {matrix[:, 0]}")


# ============================================================================
# 4. RESHAPING AND TRANSPOSING
# ============================================================================

print("\n" + "="*70)
print("4. RESHAPING AND TRANSPOSING")
print("="*70)

flat_array = np.arange(12)
print(f"\nFlat Array: {flat_array}")

reshaped = flat_array.reshape(3, 4)
print(f"\nReshaped to 3x4:\n{reshaped}")

reshaped_3d = flat_array.reshape(2, 3, 2)
print(f"\nReshaped to 2x3x2:\n{reshaped_3d}")

matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])
print(f"\nOriginal Matrix:\n{matrix}")
print(f"Transposed:\n{matrix.T}")


# ============================================================================
# 5. BROADCASTING
# ============================================================================

print("\n" + "="*70)
print("5. BROADCASTING")
print("="*70)

# Broadcasting allows operations on arrays of different shapes
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])
scalar = 10

print(f"\nMatrix:\n{matrix}")
print(f"Adding scalar {scalar}:\n{matrix + scalar}")

row_vector = np.array([1, 2, 3])
column_vector = np.array([[1], [2], [3]])

print(f"\nRow Vector: {row_vector}")
print(f"Column Vector:\n{column_vector}")
print(f"Broadcasting (Row + Column):\n{row_vector + column_vector}")


# ============================================================================
# 6. STATISTICAL FUNCTIONS
# ============================================================================

print("\n" + "="*70)
print("6. STATISTICAL FUNCTIONS")
print("="*70)

data = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])

print(f"\nData: {data}")
print(f"Mean: {np.mean(data)}")
print(f"Median: {np.median(data)}")
print(f"Std Dev: {np.std(data)}")
print(f"Variance: {np.var(data)}")
print(f"Percentiles (25, 50, 75): {np.percentile(data, [25, 50, 75])}")


# ============================================================================
# 7. RANDOM NUMBER GENERATION
# ============================================================================

print("\n" + "="*70)
print("7. RANDOM NUMBER GENERATION")
print("="*70)

np.random.seed(42)  # For reproducibility

print(f"\nRandom integers (0-10): {np.random.randint(0, 11, 5)}")
print(f"Random floats (0-1): {np.random.random(5)}")
print(f"Normal distribution: {np.random.normal(0, 1, 5)}")
print(f"Random choice from array: {np.random.choice([10, 20, 30, 40], 3)}")


# ============================================================================
# 8. BOOLEAN INDEXING AND FILTERING
# ============================================================================

print("\n" + "="*70)
print("8. BOOLEAN INDEXING AND FILTERING")
print("="*70)

data = np.array([1, 5, 3, 9, 2, 7, 4, 8, 6])

print(f"\nData: {data}")
print(f"Elements > 5: {data[data > 5]}")
print(f"Even numbers: {data[data % 2 == 0]}")
print(f"Between 3 and 7: {data[(data >= 3) & (data <= 7)]}")


print("\n" + "="*70)
print("Key Takeaways:")
print("="*70)
print("""
1. NumPy arrays are faster than Python lists
2. Broadcasting allows element-wise operations on different shapes
3. Vectorization is key to efficient numerical computing
4. Use indexing and slicing for data selection
5. Statistical functions are built-in and optimized
""")
