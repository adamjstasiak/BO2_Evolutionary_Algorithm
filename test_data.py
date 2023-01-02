import pandas as pd
import numpy as np

df = pd.read_csv(r'dataframe.csv')

df = pd.read_excel(r'test.xlsx')


df = pd.read_excel(r'flow_matrix.xlsx')
# print(df)

flow_matrix = np.zeros(df.shape)
flow = [[np.inf, 4, 2, 2, 3, 1],
        [4, np.inf, 3, 5, 5, 8],
        [2, 3, np.inf, 9, 6, 4],
        [2, 5, 9, np.inf, 7, 9],
        [3, 5, 6, 7, np.inf, 2],
        [1, 8, 4, 9, 2, np.inf]]
print(flow)
fm = df.to_numpy()
fm.T
print(fm)