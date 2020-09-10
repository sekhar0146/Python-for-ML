# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 15:42:08 2020

@author: z011348
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# heart disease analysis
heart_disease = pd.read_csv('C:/Users/z011348/Desktop/ML/input/heart-disease.csv')
heart_disease["age"].plot.hist();
#heart_disease.plot.hist(figsize=(10,30), subplots=True);

# normal mathod
over_50 = heart_disease[heart_disease["age"] > 50]
#print(len(over_50))
over_50.plot(kind='scatter',
             x='age',
             y='chol',
             c='target')

# Object oritented method mixed with pyplot
fig, ax = plt.subplots(figsize=(10,6))
over_50.plot(kind='scatter',
             x='age',
             y='chol',
             c='target',
             ax=ax)

# Object oritented method mixed from scratch
fig, ax = plt.subplots(figsize=(10,6))

sc = ax.scatter(x=over_50['age'],
                y=over_50['chol'],
                c=over_50['target'])
ax.set(title='Heart disease and cholestrol levels',
       xlabel='age',
       ylabel='Cholestrol')
ax.legend(*sc.legend_elements(), title='Target')
ax.axhline(over_50['chol'].mean(),
           linestyle='--')

#------------------
# OO - subplots
#------------------
plt.style.use('seaborn-whitegrid')
fig, (ax0,ax1) = plt.subplots(nrows=2,
                              ncols=1,
                              figsize=(10,10),
                              sharex=True)
# for ax0
sc = ax0.scatter(x=over_50['age'],
                 y=over_50['chol'],
                 c=over_50['target'],
                 cmap='winter')
ax0.set_xlim([50,80])
ax0.set(title='Heart disease and cholestrol levels',
       ylabel='Cholestrol')
ax0.legend(*sc.legend_elements(),title='Target')
ax0.axhline(over_50['chol'].mean(), linestyle='--');

# for ax1
sc = ax1.scatter(x=over_50['age'],
                 y=over_50['thalach'],
                 c=over_50['target'],
                 cmap='bwr')
ax1.set_xlim([50,80])
ax1.set_ylim([60,200])
ax1.set(title='Heart disease and max heart rate',
       xlabel='age',
       ylabel='max heart rate')
ax1.legend(*sc.legend_elements(),title='Target')
ax1.axhline(over_50['thalach'].mean(), linestyle='--');
fig.suptitle("Heart Disease Analysis", fontsize=16, fontweight='bold')
#saving the plot
fig.savefig('C:/Users/z011348/Desktop/ML/output/heart-disease-analysis.jpg')
