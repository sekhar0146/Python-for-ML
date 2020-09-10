# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 14:23:59 2020

@author: z011348
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#--------------------------
# Customizing matplotlib
#--------------------------
# print(plt.style.available)
car_sales = pd.read_csv('C:/Users/z011348/Desktop/ML/input/car-sales.csv')
car_sales['Price'] = car_sales['Price'].str.replace('[\$\,\.]', '')
car_sales['Price'] = car_sales['Price'].str[:-2].astype(int)
#print(car_sales)
car_sales['Price'].plot()
plt.style.use('seaborn-whitegrid')
car_sales['Price'].plot()
plt.style.use('seaborn')
car_sales['Price'].plot()
#car_sales.plot(x='Odometer (KM)', y=('Price'), kind='scatter');
plt.style.use('ggplot')
car_sales['Price'].plot();

# 
n = np.random.randn(10,4)
dfn = pd.DataFrame(n, columns=['a', 'b', 'c','d'])
#print(dfn)
ax = dfn.plot(kind='bar')
ax.set(title='Random bar graph from DataFrame',
       xlabel='Row number',
       ylabel='Randon number')
ax.legend().set_visible(True)
