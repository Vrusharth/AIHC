# Requirement : pip install pandas numpy seaborn matplotlib


# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('C:\\Users\\ASUS\\Sem 7 B.E\\AIHC\\EDA_of_Healthcare_data\\EDA-HEALTHCARE - EDA-HEALTHCARE.csv')

# Display first few rows of the dataset
print("First five rows of the dataset:")
print(df.head())

# Display basic information of the dataset
print("\nDataset Info:")
print(df.info())

# Display statistical summary of the dataset
print("\nStatistical Summary:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Univariate Non-Graphical Analysis
print("\nUnivariate Non-Graphical Analysis:")

# Display value counts for categorical variables
for column in df.select_dtypes(include=['object']).columns:
    print(f"\nValue counts for {column}:")
    print(df[column].value_counts())

# Multivariate Non-Graphical Analysis
print("\nMultivariate Non-Graphical Analysis:")

# Checking correlations for numerical variables
# Select only the numerical columns from the dataframe
numerical_df = df.select_dtypes(include=['float64', 'int64'])

# Now compute the correlation matrix on the numerical data only
print("Correlation matrix for numerical variables:")
print(numerical_df.corr())


# Univariate Graphical Analysis

# Plot histograms for numerical columns
print("\nUnivariate Graphical Analysis:")
df.hist(figsize=(10, 8), bins=30)
plt.suptitle('Histograms of Numerical Variables')
plt.show()

# Box plots for numerical columns to detect outliers
df.plot(kind='box', subplots=True, layout=(2, 3), figsize=(12, 8), sharex=False, sharey=False)
plt.suptitle('Boxplots of Numerical Variables')
plt.show()

# Multivariate Graphical Analysis

# Pairplot for numerical variables to understand the relationships
print("\nMultivariate Graphical Analysis:")
sns.pairplot(df, diag_kind='kde')
plt.suptitle('Pairplot for Numerical Variables', y=1.02)
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# Count plot for categorical variables
for column in df.select_dtypes(include=['object']).columns:
    plt.figure(figsize=(8, 4))
    sns.countplot(y=column, data=df)
    plt.title(f'Count plot for {column}')
    plt.show()

# Conclusion based on EDA findings
print("\nConclusion:")
# Here you can summarize insights or findings from your EDA, such as
# - The distribution of key numerical variables
# - Any evident correlations between variables
# - Presence of outliers or missing data
# - Trends observed from the multivariate graphical analysis
