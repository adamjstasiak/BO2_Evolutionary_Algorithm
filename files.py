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
