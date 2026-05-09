# run once to clean bad J data
import pandas as pd
df = pd.read_csv("dataset.csv", header=None)
df = df[df.iloc[:, 0] != 'J']
df.to_csv("dataset.csv", index=False, header=False)
print("J removed. Rows left:", len(df))