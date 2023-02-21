import pandas as pd

from modules.preprocessing import drop_col, drop_na

data = pd.read_csv("C:/Users/mkson/Downloads/data.csv", encoding="cp949")
print(len(data))
drop_na(data)
print(len(data))
