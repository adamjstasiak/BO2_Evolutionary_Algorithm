import tkinter as tk
from tkinter import ttk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from genetic_algorithm import genetic_algorithm
import function as fun
import files
import pandas as pd
from copy import copy
import dataframe_image as dfi
from PIL import Image, ImageTk


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
# print(df)
count_crossover = df.groupby('Operand type').count()

count_col = count_crossover.columns

# print(count_col)
# print(len(count_col))
count_cross = df.groupby('Crossover').count()
count_mutation = df.groupby('Mutation').count()
# print(count_crossover)
# print(count_cross)
# print(count_mutation)


idx_mut = count_mutation.index
# print(idx_mut)

dfcount = df.count()

# print(dfcount)
def genetetic_operation_analisys(path):
     cx_amount = 0
     ox_amount = 0
     pmx_amount = 0
     scramble_amount = 0
     swap_amount = 0
     inversion_amount = 0
     df = pd.read_csv(path)
     count_operand = df.groupby('Operand type').count()
     count_cross = df.groupby('Crossover').count()
     count_mutation = df.groupby('Mutation').count()
     crossover_amount = count_operand['Crossover'][0]
     mutation_amount = count_operand['Mutation'][1]
     idx_mut = count_mutation.index
     idx_cross = count_crossover.index
     for i in range(len(idx_cross)):
        if idx_cross[i] == 'CX':
             cx_amount = count_cross['Operand type'][i]
        if idx_cross[i] == 'OX':
             ox_amount = count_cross['Operand type'][i]
        if idx_cross[i] == 'PMX':
             pmx_amount = count_cross['Operand type'][i]    
     for i in range(len(idx_mut)):
        if idx_mut[i] == 'Scramble':
             scramble_amount = count_mutation['Operand type'][i] 
        if idx_mut[i] == 'Inversion':
             inversion_amount = count_mutation['Operand type'][i]
        if idx_mut[i] == 'Swap':
             swap_amount = count_mutation['Operand type'][i]
     return crossover_amount,mutation_amount,cx_amount,ox_amount,pmx_amount,inversion_amount,scramble_amount,swap_amount

# distance,flow = fun.create_random_data_matrix(6,6)

# fabric_list = fun.create_fabric_list(6)
# population = fun.generate_population(fabric_list,20)

# selection = fun.ranking_selection(population,distance,flow,50)

# print(selection)
# print(len(selection))
# best_individual, current_min_value, min_values_list, operand_type, crossover_type, mutation_type = genetic_algorithm(distance,flow,fabric_list, 40,20, 50,crossover_probability=70,mutation_probability=30,pmx_probability=0,cx_probability=1,ox_probability=0)
df_crossover = pd.read_csv('mutation_value.csv')

df_group = df_crossover.groupby('Operand').min()

# print(df_group)