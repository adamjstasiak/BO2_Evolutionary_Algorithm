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