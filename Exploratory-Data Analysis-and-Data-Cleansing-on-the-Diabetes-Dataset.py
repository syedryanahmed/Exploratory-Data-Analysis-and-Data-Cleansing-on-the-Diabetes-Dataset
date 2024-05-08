import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Task 1. Read the csv data into a data frame
df = pd.read_csv('data/diabetes.csv')

# Task 2. Print the first 10 rows
print(df.head(10))

# Task 3. Print data types, column names, null value counts, and memory consumption
print("\nData Types:")
print(df.dtypes)

print("\nColumn Names:")
print(df.columns)

print("\nNull Value Counts:")
print(df.isnull().sum())

print("\nMemory Consumption:")
print(df.memory_usage(deep=True))

# Task 4. Print basic statistical details
print("\nSummary Statistics:")
print(df.describe())

# Task 5. Print statistical details by transposing the data frame
print("\nTransposed Summary Statistics:")
df_transposed = df.transpose()
print(df_transposed.describe())

# Task 6. Replace 0 with nan for specified columns
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[zero_cols] = df[zero_cols].replace(0, np.nan)

# Task 7. Plot data distribution for each column
print("\nData Distribution before Imputation:")
for col in df.columns:
    plt.figure()
    df[col].hist()
    plt.title(f'Distribution of {col}')
    plt.show()

# Task 8. Fill missing values with right strategies
df['Glucose'] = df['Glucose'].fillna(df['Glucose'].median())
df['BloodPressure'] = df['BloodPressure'].fillna(df['BloodPressure'].median())
df['SkinThickness'] = df['SkinThickness'].fillna(df['SkinThickness'].median())
df['Insulin'] = df['Insulin'].fillna(df['Insulin'].median())
df['BMI'] = df['BMI'].fillna(df['BMI'].median())

# Task 9. Plot distribution after filling missing data
print("\nData Distribution after Imputation:")
for col in df.columns:
    plt.figure()
    df[col].hist()
    plt.title(f'Distribution of {col} after filling missing values')
    plt.show()