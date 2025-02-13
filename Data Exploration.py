import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("~/Downloads/secondary_data.csv", sep=';')

print(df.shape[0])
print(df.shape[1])


# plotting the outcome variable
outcome_var = df.columns[-0]
plt.figure(figsize=(8, 6))
df[outcome_var].value_counts().plot(kind='bar', color='skyblue')
plt.xlabel('Poisonous state of mushroom')
plt.ylabel('Count')
plt.title('Distribution of Outcome Variable')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Checking for null values
missing_percentage = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({'Variable': df.columns, '% Missing': missing_percentage})

print(missing_df)