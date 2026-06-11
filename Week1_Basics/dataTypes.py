import pandas as pd

data = {
    'Name': ['Alice', 'Bob', 'Charlie'],        # Nominal
    'Age': [25, 30, 22],                         # Continuous
    'Purchases': [3, 7, 1],                      # Discrete
    'Rating': ['Good', 'Average', 'Excellent'],  # Ordinal
    'City': ['Delhi', 'Mumbai', 'Delhi']         # Nominal
}
df = pd.DataFrame(data)
print(df.dtypes)   # see how pandas reads each column
