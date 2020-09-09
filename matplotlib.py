import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.plot()
#plt.show() # show and plot are same
plt.plot([1,2,3,9])

x = [1,3,5,7]
y = [11,44,66,88]
y2 = [11,33,55,77]
plt.plot(x,y)

# method 1 to create plots
fig = plt.figure()
ax = fig.add_subplot()
plt.show()

# method 2 to create plots
fig = plt.figure()
ax = fig.add_axes([1,1,1,1])
plt.show()

fig = plt.figure()
ax = fig.add_axes([1,1,1,1])
ax.plot(x,y)
plt.show()

# method 3 to create plots (Recommanded)
fig, ax = plt.subplots()
ax.plot(x,y)

########################################
# simple plot
########################################
x1 = [2,4,6,8]
y1 = [33,55,44,99]

fig, ax = plt.subplots(figsize=(10,10))

ax.plot(x1,y1)

ax.set(title="Sample plot", xlabel='x-values', ylabel='y-values')

fig.savefig('C:/Users/z011348/Desktop/ML/output/sample_plot.jpg')

########################################
# Plots with numpy
########################################
x_array = np.linspace(1,10,100)
print(x_array)
# plot the data - Line plot
fig, ax = plt.subplots()
ax.plot(x_array, x_array**2)

# scatter plot
fig, ax = plt.subplots()
ax.scatter(x_array, np.exp(x_array));

fig, ax = plt.subplots()
ax.scatter(x_array, np.sin(x_array));

# Bar chart
nut_butter_prices = {"almond butter":10,
                     "panut butter":8,
                     "cashew butter":12}
fig, ax = plt.subplots()
ax.bar(nut_butter_prices.keys(), nut_butter_prices.values())
ax.set(title='Nut butter prices', ylabel='prices')

# barh chart
fig, ax = plt.subplots()
ax.barh(list(nut_butter_prices.keys()), list(nut_butter_prices.values()))
ax.set(title='Nut butter prices', xlabel='prices($)')

# histogram
np.random.seed(4)
x = np.random.randn(100)
fig, ax = plt.subplots()
ax.hist(x)

# Multiple plots
fig, ((ax1,ax2), (ax3,ax4)) = plt.subplots(nrows=2,
                                           ncols=2,
                                           figsize=(10,5))
#- plot for each for axis
x1=np.random.randn(50)
ax1.plot(x1, x1/2)
ax2.scatter(np.random.random(10), np.random.random(10))
ax3.bar(nut_butter_prices.keys(), nut_butter_prices.values())
ax4.hist(np.random.randn(100))

# Multip+le plots - specifically axes 
fig, ax = plt.subplots(nrows=2,
                       ncols=2,
                       figsize=(10,5))
#-- plot for each for axis
x1=np.random.randn(20)
ax[0,0].plot(x1, x1/2)
ax[0,1].scatter(np.random.random(20), np.random.random(20))
ax[1,0].bar(nut_butter_prices.keys(), nut_butter_prices.values())
ax[1,1].hist(np.random.randn(500))

########################################
# Plots with Pandas
########################################
car_sales = pd.read_csv('C:/Users/z011348/Desktop/ML/input/car-sales.csv')
print(car_sales)
car_sales['Price'] = car_sales['Price'].str.replace('[\$\,\.]', '')
car_sales['Price'] = car_sales['Price'].str[:-2].astype(int)
car_sales['Sales_date'] = pd.date_range("7/1/2020", periods=len(car_sales))
car_sales['Total_sales'] = car_sales['Price'].cumsum()
print(car_sales)
# ploting data with DF
car_sales.sort_values('Price',
                      ascending = True,
                      inplace = True)
car_sales.plot(x='Sales_date',y='Total_sales')
car_sales.plot(x='Odometer (KM)', y='Price', kind='scatter')
car_sales.plot(x='Make', y='Price', kind='bar')

#
x = np.random.rand(10,4)
df = pd.DataFrame(x,columns=['A','B','C','D'])
df.plot.bar()
df.plot(kind='bar')

# hist
#print(car_sales['Price'].plot(kind='hist'))
car_sales.plot(x='Make', y='Price', kind='hist')
