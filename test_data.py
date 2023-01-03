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


df = pd.read_csv('genetics_operation.csv')
print(df)
count_crossover = df.groupby('Operand type').count()

count_cross = df.groupby('Crossover').count()
count_mutation = df.groupby('Mutation').count()
print(count_crossover)
print(count_cross)
print(count_mutation)


def genetetic_operation_analisys(path):
     df = pd.read_csv(path)
     count_crossover = df.groupby('Operand type').count()
     count_cross = df.groupby('Crossover').count()
     count_mutation = df.groupby('Mutation').count()
     crossover_amount = count_crossover['Crossover'][0]
     mutation_amount = count_crossover['Mutation'][1]
     cx_amount = count_cross['Operand type'][0]
     ox_amount = count_cross['Operand type'][1]
     pmx_amount = count_cross['Operand type'][2]
     inversion_amount = count_mutation['Operand type'][0]
     scramble_amount = count_mutation['Operand type'][1]
     swap_amount = count_mutation['Operand type'][2]
     return crossover_amount,mutation_amount,cx_amount,ox_amount,pmx_amount,inversion_amount,scramble_amount,swap_amount

