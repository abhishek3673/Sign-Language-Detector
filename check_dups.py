import pandas as pd
df = pd.read_csv("dataset.csv", header=None)
print(df.iloc[:, 0].value_counts())
print("Total rows:", len(df))