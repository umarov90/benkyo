import pandas as pd
import sys

table = pd.read_csv(sys.argv[1], sep="\t", index_col=None, header=None)
table1 = table.iloc[:, :2]
table1 = table1.rename(columns={0: 'chrom', 1: 'start'})
table2 = table.iloc[:, 3:5]
table2 = table2.rename(columns={3: 'chrom', 4: 'start'})
table = pd.concat([table1, table2], ignore_index=True)
table["start"] = table["start"] // 10000
table = table.groupby(['chrom', 'start']).size().to_frame(name = 'size').reset_index()
table.head(5)