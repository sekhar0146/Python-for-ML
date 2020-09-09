# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 11:02:18 2020

@author: z011348
"""
import numpy as np
import pandas as pd
import timeit

a1 = np.array([7,8,9])
print("numpy array a1 : ", a1)
print("Type of array : ", type(a1)) # <class 'numpy.ndarray'>

a2 = np.array([
    [1.,2.9,3.4],
    [5.,7.,9.]
    ])
print("numpy array A2 : ", a2)
print("Type of array a2 : ", type(a2)) # <class 'numpy.ndarray'>

a3 = np.array([
    [[1,2,3],
     [4,5,6],
     [7,8,9]],
    [[10,11,12],
     [13,14,15],
     [16,17,18]]
    ])
print("numpy array A3 : ")
print(a3)
print(a3.shape)
print(a3.ndim)  # 3
print(a1.ndim)  # 1
print(a2.dtype)
print(a3.size) # 18
print(a2.size) # 6

print(" ==== array Data - Begin ===== ")
print(a3[0])
print(a3[1])
print(a2[0])
print(" ==== array Data - Fin ===== ")
# use numpy array in Dataframe
df = pd.DataFrame(a2)
print(df)

# Creating numpy arrays

arr1 = np.array([1,2,3])
print(arr1)
print('')
arrones = np.ones((2,2))
print(arrones)
print('')
arrzeros = np.zeros((2,2))
print(arrzeros)
print('')
nparange = np.arange(0,10,2)
print(nparange)
print('')
# create random arrays  - data will keep on changing 
random_arr = np.random.random(size=(2,2))
print(random_arr)
print('')
random_array = np.random.randint(0,10, size=(3,3))
print(random_array)
print('')
# random arrays with seed - data will not changing
np.random.seed(0)
random_seed = np.random.randint(0,10, size=(3,3)) 
print(random_seed)
print('')
npunique = np.unique(random_seed)
print(npunique)

print('===============================')
print(a3)
print(a3[:1, :1, :2])
print(a3)
print(a3[:1, :2, :2])
print(a3)
print(a3[:2, :2, :2])
print(a3)
print(a3[:2, 1:3, :2])
print(a3)
print(a3[:2, 1:3, 1:3])
print(a3)
print(a3[:, :, :1])
print('')
# ====== Manipulating arrays
print('a1 values:')
print(a1)
print('a2 values:')
print(a2)
print('')
print(a1 + 1)
print('')
print(a1 * 2)
print('')
print(a1 * a2)
print('')
print(a1 / a2)
print(a1 // a2) # round the result
print('')
print(a2 ** 2)
print(np.square(a2)) # same as above
print('')

print(np.sum(a2))
print(np.sum(a3)) # np.sum() is faster than python sum
print(np.mean(a1))
print(np.average(a2))
print(np.std(a2)) # std = sqrt of variance
print(np.var(a2))
print(np.sqrt(np.var(a2)))

## reshaping and transpose 
print('==== reshaping and transpose =====')
# a2*a3 - operands could not be broadcast together with shapes (2,3) (2,3,3) 
print(a2)
print(a2.shape)
a2_reshape = a2.reshape(2,3,1)
print(a2_reshape.shape)
print(a2_reshape * a3)
print('')
print(a2)
print(a2.T) # T = Transpose 
print(a3)
print(a3.T)
print('')
# dot product 
print('dot product')
mat1 = np.random.randint(10, size=(3,2))
mat2 = np.random.randint(10, size=(3,2))
print(mat1)
print(mat2)
print('')
# dot product is not possible as inner axis is not same
# (3,2) * (3,2) : mat1 colums should match with rows in mat2 
print(mat2.shape)
print(mat1.T)
print(mat1.T.shape)
print('mat1 transpose')
print(mat1.T)
print('mat2')
print(mat2)
print('Dot product result: - ')
mat_dot = np.dot(mat1.T,mat2)
print(mat_dot)
print(mat_dot.shape)
print('')
# comparison

print(a1)
print(a2)
print(a1 == a2)
print(a1 > a2)
array_comp = (a1 >= a2)
print(array_comp)
print(array_comp.dtype)
print('')
# sorting arrays
print('sorting arrays')
ran_array = np.random.randint(10, size=(5,3))
print(ran_array)
print(np.sort(ran_array))

# pandas image
from matplotlib.image import imread
img = imread('C:/Users/z011348/Pictures/panda.jpg')
print(type(img))
print(img)
print(img.size, img.shape, img.ndim)