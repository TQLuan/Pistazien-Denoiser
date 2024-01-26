# -*- coding: utf-8 -*-
"""
Plot the evaluation scores (Gain, STOI) in different noise level

Usage:
    ./Plot/plot.py

Author:
    Tianqi Luan
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


df1 = pd.read_csv('./results0.csv')
df1['Noise Level'] = '0 dB'
df2 = pd.read_csv('./results5.csv')
df2['Noise Level'] = '5 dB'
df3 = pd.read_csv('./results10.csv')
df3['Noise Level'] = '10 dB'
df4 = pd.read_csv('./results15.csv')
df4['Noise Level'] = '15 dB'
df5 = pd.read_csv('./results20.csv')
df5['Noise Level'] = '20 dB'
df6 = pd.read_csv('./results25.csv')
df6['Noise Level'] = '25 dB'
df7 = pd.read_csv('./results30.csv')
df7['Noise Level'] = '30 dB'

method_mapping = {
    'logmnse': 'MMSE-LSA',
    'mmse': 'MMSE-STSA',
    'dtln': 'DTLN',
    'garch': 'DTLN+Garch',
    'demucs': 'Demucs'
}
print(df1['Method'].unique())

df1['Method'] = df1['Method'].map(method_mapping)
df2['Method'] = df2['Method'].map(method_mapping)
df3['Method'] = df3['Method'].map(method_mapping)
df4['Method'] = df4['Method'].map(method_mapping)
df5['Method'] = df5['Method'].map(method_mapping)
df6['Method'] = df6['Method'].map(method_mapping)
df7['Method'] = df7['Method'].map(method_mapping)
print(df1['Method'].unique())

datasets = {'0 dB': df1, '5 dB': df2, '10 dB': df3, '15 dB': df4, '20 dB': df5, '25 dB': df6, '30 dB': df7}

merged_data = pd.DataFrame()


df_list = [df1, df2, df3, df4, df5, df6, df7]


for df in df_list:
    merged_data = pd.concat([merged_data, df], ignore_index=True)


data = merged_data
data = data[data['Method'] != 'MMSE-LSA']

grouped_data = data.groupby(['Noise Level', 'Method'])['STOI'].describe()

if __name__ == "__main__":
    sns.boxplot(x='Method', y='STOI', hue='Noise Level', data=data)
    plt.title("STOI Comparison for Different Methods and Noise Levels")
    plt.xlabel("Method")
    plt.ylabel("STOI")
    plt.grid()
    plt.legend(ncol=3, title='Noise Level', fontsize='small')
    plt.show()
