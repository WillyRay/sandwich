import pandas as pd

df = pd.read_csv("./data/cdiff.csv")

counts = df['studyday'].value_counts()
counts = counts.sort_index()
counts = counts.reindex(range(0,56), fill_value=0)
print(counts)

counts.to_csv("./data/surfaces_by_day.csv")

df2 = pd.read_csv("./data/simulated_base2.csv", index_col=[2,3,4])
print(df2)
df2.to_csv('./data/surfaces_indexed.csv')