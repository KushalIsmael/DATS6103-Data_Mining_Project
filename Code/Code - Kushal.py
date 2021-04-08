import pandas as pd
import os

path = str(os.getcwd())
print(path)
data = pd.read_csv(path[:-4]+"nonvoters_data.csv")

data.head(10)