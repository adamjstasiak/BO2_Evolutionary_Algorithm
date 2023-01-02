#!/usr/bin/python
# -*- coding: utf-8 -*-
import csv
import pandas as pd


def export_to_csv(data, filename):
    file = open(filename, 'w')
    writer = csv.writer(file)
    dataframe = [['index', 'value']]
    i=1
    for el in data:
        dataframe.append([i,el])
        i += 1
    writer.writerows(dataframe)
    file.close()


def clearing_csv(filename):
    f = open(filename, 'w+')
    f.close()
