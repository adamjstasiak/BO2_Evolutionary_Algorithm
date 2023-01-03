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
    cx_amount = 0
    ox_amount = 0
    pmx_amount = 0
    swap_amount = 0
    scramble_amount = 0
    inversion_amount = 0
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

# crossover_amount,mutation_amount,cx_amount,ox_amount,pmx_amount,inversion_amount,scramble_amount,swap_amount = genetetic_operation_analisys('genetics_operation.csv')
