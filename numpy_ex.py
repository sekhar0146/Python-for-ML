# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 11:09:59 2020

@author: z011348
"""
import pandas as pd
import numpy as np

# =============================================================================
# import csv file from a path
# =============================================================================
butter_sales = pd.read_csv('C:/Users/z011348/Desktop/ML/input/butter_sales.csv',
                           index_col=0)
print(butter_sales)
print(butter_sales.shape)

butter_prices = pd.read_csv('C:/Users/z011348/Desktop/ML/input/butter_prices.csv',
                           index_col=0)
print(butter_prices)
print(butter_prices.shape)
#print(np.dot(butter_sales,butter_prices.T))
daily_sales = np.dot(butter_sales,butter_prices.T)
butter_sales["Total_sales($)"] =  daily_sales
print(butter_sales)
print('')
# assignments
print('assignments')

onedm = np.array([4,7,2])
print('onedim array :', onedm)
print('shape :', onedm.shape)
print('dimention :', onedm.ndim)
print('')
twodm = np.array([[4,7,2],[67,34,90]])
print('twodim array :')
print(twodm)
print('shape :', twodm.shape)
print('dimention :', twodm.ndim)
print('')
threedm = np.array([
    [[10,4,2],
    [67,34,10],
    [6,49,6]],
    
    [[4,1,2],
    [67,0,30],
    [9,2,6]]
    ])
print('threedim array :')
print(threedm)
print('shape :', threedm.shape)
print('dimention :', threedm.ndim)
print('size :', threedm.size)   
print('Data Type :', threedm.dtype) 
print('Class Type :', type(threedm))
print('')

print('convert numpy array into DataFrame')
twodm_DF = pd.DataFrame(twodm)
print(twodm_DF)      
onedm_DF = pd.DataFrame(onedm)
print(onedm_DF)
print('')
# Create an array of shape (10, 2) with only ones
print('# Create an array of shape (10, 2) with only ones')
ones = np.ones(shape=(10,2))
print(ones)
# Create an array of shape (7, 2, 3) of only zeros
print('# Create an array of shape (7, 2, 3) of only zeros')
zeros = np.zeros(shape=(7,2,3))
print(zeros)
print('')
print('# Create an array within a range of 0 and 100')
onedm_range = np.arange(0, 100, 3)
print(onedm_range)
print(onedm_range.ndim)
print('')
print('# Create a random array with numbers between 0 and 10 of size (7, 2):')
ran_array = np.random.randint(10, size=(7,2))
print(ran_array)
print(ran_array.ndim)
print('')
# Create a random array of floats between 0 & 1 of shape (3, 5)
print('# Create a random array of floats between 0 & 1 of shape (3, 5) :')
ran_array_float = np.random.random(size=(3,5))
print(ran_array_float)
print('')
# Create a random array of numbers between 0 & 10 of size (4, 6)
# set random seed
print('random seed')
np.random.seed(0)
ran_seed = np.random.randint(0,10, size=(4,6))
print(ran_seed)
# Find the unique numbers in the array just created
print('unique numbers:')
print(np.unique(ran_seed))
# Find the 0'th index of the latest array you created
print('')
print('# Find the 0th index of the latest array created: ')
print(ran_seed[0]) 
print('# Get the first 2 rows')
print(ran_seed[:2])
print('# Get only 2nd row')
print(ran_seed[1:2])
print('# Get the first 2 values of the first 2 rows of the latest array:')
print(ran_seed[:2,:2])      
print('')
print("""
# Create a random array of numbers between 0 & 10 and an array of ones
# both of size (3, 5), save them both to variables
# Add the two arrays together
# Create another array of ones of shape (5, 3)
# Try add the array of ones and the other most recent array together
      """)
rand_10 = np.random.randint(0,10, size=(3,5))
rand_1 = np.ones(shape=(3,5))
print(rand_10)
print(rand_1)
print(rand_10 + rand_1)
rand_11 = np.ones(shape=(5,3))
print(rand_1 + rand_11.T)
print('')
print("""
# Create another array of ones of shape (3, 5)
# Subtract the new array of ones from the other most recent array
# Multiply the ones array with the latest array
# Take the latest array to the power of 2 using '**'
# Do the same thing with np.square()
# Find the mean of the latest array using np.mean()
# Find the maximum of the latest array using np.max()
# Find the minimum of the latest array using np.min()
# Find the standard deviation of the latest array
# Find the variance of the latest array
# Reshape the latest array to (3, 5, 1)
# Transpose the latest array
      """)
print(rand_10)
ones_arr = np.ones(shape=(3,5))
print(ones_arr)
print(ones_arr - rand_10)
print(ones_arr * rand_10)
print(rand_10 ** 2) # power 
print(np.square(rand_10)) # same as power
print(np.mean(rand_10))
print(np.max(rand_10))
print(np.min(rand_10))
print(np.std(rand_10))
print(np.var(rand_10))
reshape_array=rand_10.reshape(3,5,1)
print(reshape_array)
array_t = reshape_array.T
print(array_t)
print('')

print("""
# Create two arrays of random integers between 0 to 10
# one of size (3, 3) the other of size (3, 2)
# Perform a dot product on the two newest arrays you created
      """)
r1 = np.random.randint(0,10, size=(3,3))
r2 = np.random.randint(0,10, size=(3,2))
print(r1)
print(r2)
rdot = r1.dot(r2)
print(rdot)

print("""
# Create two arrays of random integers between 0 to 10
# both of size (4, 3)
# Perform a dot product on the two newest arrays you created
      """) 
rd1 = np.random.randint(0,10, size=(4,3))
rd2 = np.random.randint(0,10, size=(4,3))
print(rd1)
print(rd2)
rddot = rd1.dot(rd2.T)
print(rddot)

print("# Compare the two arrays with '>' :")
print(rd1 > rd2)
print("# Compare the two arrays with '>=' " )
print(rd1 >= rd2)
print("# Find which elements of the first array are greater than 7 :")
print(rd1 > 7)
print("# Which parts of each array are equal? (try using '==') : ")
print(rd1 == rd2)
print("# Sort one of the arrays you just created in ascending order :")
print(rd1)
print(np.sort(rd1))
print("# Sort the indexes of one of the arrays you just created")
print(np.argsort(rd1))
print("# Create an array with 10 evenly spaced numbers between 1 and 100 :")
print(np.linspace(1, 100, 10))