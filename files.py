#!/usr/bin/python
# -*- coding: utf-8 -*-
import csv
import xml
import pandas as pd
from math import inf 

def import_flow_matrix(path):
    df = pd.read_excel(path)
    matrix = df.to_numpy()
    matrix.T
    return matrix

def export_to_csv_values(data, filename):
    file = open(filename, 'w')
    writer = csv.writer(file)
    dataframe = [['index', 'value']]
    i=1
    for el in data:
        dataframe.append([i,el])
        i += 1
    writer.writerows(dataframe)
    file.close()

def export_to_csv_characteristics(data,filename):
    file = open(filename, 'w')
    writer = csv.writer(file)
    dataframe = [['Operand type', 'Crossover','Mutation']]
    for el in range(len(data[0])):
        dataframe.append([data[0][el],data[1][el],data[2][el]])
    writer.writerows(dataframe)
    file.close()

def export_to_csv_operand_values(data,filename):
    file = open(filename, 'w')
    writer = csv.writer(file)
    dataframe = [['Operand','Value','Delta']]
    for el in range(len(data)):
        dataframe.append([data[el][0],data[el][1],data[el][2]])
    writer.writerows(dataframe)
    file.close()

def clearing_csv(filename):
    f = open(filename, 'w+')
    f.close()


def genetetic_operation_analisys(path):
    df = pd.read_csv(path)
    count_operand = df.groupby('Operand type').count()
    count_cross = df.groupby('Crossover').count()
    count_mutation = df.groupby('Mutation').count()
    crossover_amount = count_operand['Crossover'][0]
    mutation_amount = count_operand['Mutation'][1]
    idx_mut = count_mutation.index
    idx_cross = count_cross.index
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

def genetics_operation_analisys(path_1,path_2):
    df_crossover = pd.read_csv(path_1)
    df_mutation = pd.read_csv(path_2)
    df_crossover_mean = df_crossover.groupby('Operand').mean()
    df_mutation_mean = df_mutation.groupby('Operand').mean()
    mutation_value = []
    crossover_value = []
    mutation_delta = []
    crossover_delta = []
    cx_value = 0
    ox_value = 0
    pmx_value = 0
    cx_delta = 0
    ox_delta = 0
    pmx_delta = 0
    swap_value  = 0
    scramble_value = 0
    inversion_value = 0
    swap_delta  = 0
    scramble_delta = 0
    inversion_delta = 0
    idx_mut = df_mutation_mean.index
    idx_cross = df_crossover_mean.index
    for i in range(len(idx_cross)):
        if idx_cross[i] == 'CX':
            cx_value = df_crossover_mean['Value'][i]
            cx_delta = df_crossover_mean['Delta'][i]
        if idx_cross[i] == 'OX':
            ox_value = df_crossover_mean['Value'][i]
            ox_delta = df_crossover_mean['Delta'][i]
            
        if idx_cross[i] == 'PMX':
            pmx_value = df_crossover_mean['Value'][i]
            pmx_delta = df_crossover_mean['Delta'][i]
            
    for i in range(len(idx_mut)):
        if idx_mut[i] == 'Scramble':
            scramble_value = df_mutation_mean['Value'][i]
            scramble_delta = df_crossover_mean['Delta'][i]
            
        if idx_mut[i] == 'Inversion':
            inversion_value = df_mutation_mean['Value'][i]
            inversion_delta = df_crossover_mean['Delta'][i]
            
        if idx_mut[i] == 'Swap':
            swap_value = df_mutation_mean['Value'][i]
            swap_delta = df_crossover_mean['Delta'][i]
            
        crossover_value.append(('CX',cx_value))
        crossover_delta.append(('CX',cx_delta))
        crossover_value.append(('OX',ox_value))
        crossover_delta.append(('OX',ox_delta))
        crossover_value.append(('PMX',pmx_value))
        crossover_delta.append(('PX',pmx_delta))
        mutation_value.append(('Scramble',scramble_value))
        mutation_delta.append(('Scramble',scramble_delta))
        mutation_value.append(('Inversion',inversion_value))
        mutation_delta.append(('Inversion',inversion_delta))
        mutation_value.append(('Swap',swap_value))
        mutation_delta.append(('Swap',swap_delta))  
    return 



