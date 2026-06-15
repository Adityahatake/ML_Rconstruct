"""
==============================================================================
MACHINE LEARNING DATA TYPES - COMPREHENSIVE GUIDE
==============================================================================

DATA TYPES IN ML:
1. NOMINAL (Categorical) - No inherent order or ranking
2. ORDINAL (Categorical) - Has a meaningful order/hierarchy
3. DISCRETE (Quantitative) - Countable, whole numbers only
4. CONTINUOUS (Quantitative) - Infinite values in a range

This script demonstrates each data type with real-world examples.
==============================================================================
"""

import pandas as pd

# Create a sample dataset with different data types
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],        # NOMINAL: Names (no order)
    'Age': [25, 30, 22],                         # CONTINUOUS: Can be 25.5, 29.3, etc.
    'Purchases': [3, 7, 1],                      # DISCRETE: Whole numbers only (can't buy 3.5 items)
    'Rating': ['Good', 'Average', 'Excellent'],  # ORDINAL: Has ranking (Excellent > Good > Average)
    'City': ['Delhi', 'Mumbai', 'Delhi']         # NOMINAL: City names (no order)
}

# Convert dictionary to pandas DataFrame for analysis
df = pd.DataFrame(data)

print("\n" + "="*80)
print("DATA TYPES - RECOGNIZED BY PANDAS")
print("="*80)
print(df.dtypes)   # Shows how pandas infers each column's data type
print("Note: object type = string/text, int64 = integer numbers\n")

print("="*80)
print("DETAILED DATA TYPE EXPLANATIONS & EXAMPLES")
print("="*80)

print("\n[1] NOMINAL - Categorical data with NO order/ranking")
print("-" * 80)
print("  Definition: Categories that cannot be ranked or ordered meaningfully")
print("  Examples in our data:")
print("    - Name column: Alice, Bob, Charlie (can't rank names)")
print("    - City column: Delhi, Mumbai, Delhi (cities have no inherent order)")
print("  Real-world examples: Colors, Countries, Gender, Product brands")
print("  ✗ You CANNOT say: 'Alice > Bob' or 'Delhi > Mumbai'")

print("\n[2] ORDINAL - Categorical data WITH meaningful order")
print("-" * 80)
print("  Definition: Categories with a clear hierarchy or ranking")
print("  Examples in our data:")
print("    - Rating column: Good, Average, Excellent")
print("    - Ranking order: Excellent (highest) > Good > Average (lowest)")
print("  Real-world examples: Education level (High School < Bachelor < Master),")
print("                      Satisfaction (Poor < Fair < Good < Excellent)")
print("  ✓ You CAN say: 'Excellent > Good > Average'")

print("\n[3] DISCRETE - Quantitative data with COUNTABLE values")
print("-" * 80)
print("  Definition: Whole numbers only, finite and distinct values")
print("  Examples in our data:")
print("    - Purchases column: 3, 7, 1 (number of items purchased)")
print("  Real-world examples: Number of students, Goals scored, Items in stock")
print("  Characteristics:")
print("    • Can only take specific values (1, 2, 3... not 3.5)")
print("    • Often used for counting")
print("    • Has gaps between values")

print("\n[4] CONTINUOUS - Quantitative data with INFINITE values")
print("-" * 80)
print("  Definition: Any value within a range, can have decimals")
print("  Examples in our data:")
print("    - Age column: 25, 30, 22 (but could be 25.5, 29.7, 22.3...)")
print("  Real-world examples: Height, Weight, Temperature, Distance, Time")
print("  Characteristics:")
print("    • Can take ANY value in a range (including decimals)")
print("    • No gaps between values")
print("    • Infinite precision possible")

print("\n" + "="*80)
print("QUICK REFERENCE TABLE")
print("="*80)
print(f"{'Type':<12} {'Category':<15} {'Order':<10} {'Example':<20}")
print("-" * 80)
print(f"{'Nominal':<12} {'Categorical':<15} {'No':<10} {'City, Gender':<20}")
print(f"{'Ordinal':<12} {'Categorical':<15} {'Yes':<10} {'Rating, Level':<20}")
print(f"{'Discrete':<12} {'Quantitative':<15} {'N/A':<10} {'Count, Items':<20}")
print(f"{'Continuous':<12} {'Quantitative':<15} {'N/A':<10} {'Age, Height':<20}")
print("="*80)
