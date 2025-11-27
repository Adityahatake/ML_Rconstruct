import pandas as pd

# pandas_begin.py
# Introduction to pandas for beginners

# What is pandas?
# pandas is a powerful Python library for data analysis and manipulation.
# It provides easy-to-use data structures and functions for working with structured data.

# How to install pandas:
# Run the following command in your terminal or command prompt:
# pip install pandas

# Importing pandas


# 1. Creating Data Structures


# 1.1 Series: One-dimensional labeled array

data = [10, 20, 30, 40]
series = pd.Series(data)
print("Series:")
print(series)

# You can specify custom labels (index)

labels = ['a', 'b', 'c', 'd']
series_with_labels = pd.Series(data, index=labels)
print("\nSeries with custom labels:")
print(series_with_labels)

# 1.2 DataFrame: Two-dimensional labeled data structure (like a table)
data_dict = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'Paris', 'London']
}
df = pd.DataFrame(data_dict)
print("\nDataFrame:")
print(df)

# 2. Reading Data

# 2.1 Reading from a CSV file
# df = pd.read_csv('filename.csv')

# 2.2 Reading from an Excel file
# df = pd.read_excel('filename.xlsx')

# 3. Basic DataFrame Operations

# 3.1 Viewing data
print("\nFirst 2 rows:")
print(df.head(2))  # First 2 rows

print("\nLast row:")
print(df.tail(1))  # Last row

# 3.2 Getting info about DataFrame
print("\nInfo about DataFrame:")
print(df.info())

print("\nSummary statistics:")
print(df.describe())

# 3.3 Selecting columns
print("\nSelecting 'Name' column:")
print(df['Name'])

# 3.4 Selecting rows by index
print("\nSelecting first row:")
print(df.iloc[0])

# 3.5 Selecting rows by label
print("\nSelecting rows where Age > 28:")
print(df[df['Age'] > 28])

# 4. Modifying Data

# 4.1 Adding a new column
df['Country'] = ['USA', 'France', 'UK']
print("\nDataFrame with new column:")
print(df)

# 4.2 Changing values
df.at[1, 'Age'] = 31  # Change Bob's age to 31
print("\nDataFrame after changing Bob's age:")
print(df)

# 4.3 Removing columns
df = df.drop('Country', axis=1)
print("\nDataFrame after removing 'Country' column:")
print(df)

# 4.4 Removing rows
df = df.drop(2)  # Remove row with index 2 (Charlie)
print("\nDataFrame after removing Charlie:")
print(df)

# 5. Handling Missing Data

# Create DataFrame with missing values
data_with_nan = {
    'A': [1, 2, None],
    'B': [4, None, 6]
}
df_nan = pd.DataFrame(data_with_nan)
print("\nDataFrame with missing values:")
print(df_nan)

# 5.1 Detect missing values
print("\nIs null:")
print(df_nan.isnull())

# 5.2 Fill missing values
df_filled = df_nan.fillna(0)
print("\nFilled missing values with 0:")
print(df_filled)

# 5.3 Drop rows with missing values
df_dropped = df_nan.dropna()
print("\nDropped rows with missing values:")
print(df_dropped)

# 6. Sorting Data

# 6.1 Sort by column values
print("\nSort by Age:")
print(df.sort_values('Age'))

# 7. Grouping Data

# Create a new DataFrame for grouping
group_data = {
    'Team': ['A', 'A', 'B', 'B'],
    'Points': [10, 15, 20, 25]
    
}
group_df = pd.DataFrame(group_data)
print("\nGroup DataFrame:")
print(group_df)

# 7.1 Group by 'Team' and sum points
grouped = group_df.groupby('Team')['Points'].sum()
print("\nGrouped by Team and summed Points:")
print(grouped)

# 8. Applying Functions

# 8.1 Apply a function to a column
df['Age_plus_10'] = df['Age'].apply(lambda x: x + 10)
print("\nDataFrame with Age plus 10:")

print(df)

# 9. Saving Data

# 9.1 Save DataFrame to CSV
# df.to_csv('output.csv', index=False)

# 9.2 Save DataFrame to Excel
# df.to_excel('output.xlsx', index=False)

# 10. More Resources

# - Official pandas documentation: https://pandas.pydata.org/docs/
# - pandas tutorials: https://pandas.pydata.org/pandas-docs/stable/getting_started/intro_tutorials/index.html

# This script covers the basics of pandas.
# Try modifying the code and experimenting with your own data!