import pandas as pd
df = pd.read_csv("dataset.csv", header=None, low_memory=False)

# find which rows are the "9" with 48 cases
# since 9 appears twice, one is string "9" and other is int 9
# convert all to string first
df.iloc[:, 0] = df.iloc[:, 0].astype(str).str.strip()

# check what variants exist
print(df.iloc[:, 0].value_counts())