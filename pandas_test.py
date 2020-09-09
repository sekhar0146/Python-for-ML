
import pandas as pd
import matplotlib as plt

import sklearn

print('The scikit-learn version is {}.'.format(sklearn.__version__))

colour = pd.Series(['white', 'blue', 'green'])
print(colour)
car_type = pd.Series(['Renault', 'Nissan', 'Toyota'])
print(car_type)
car = pd.DataFrame(
    {
     'car_brand': car_type,
     'car_colour': colour
     }
    )
print(car)
print('')
# =============================================================================
# import csv file from a path
# =============================================================================
car_sales = pd.read_csv('C:/Users/z011348/Desktop/ML/input/car-sales.csv')
print(car_sales)
print('')
print(car_sales.dtypes)
print('')
print(car_sales.describe)
print('')
print(car_sales.info())
print('')
print(car_sales.columns)
print(len(car_sales))
print(car_sales.head(7))
print(car_sales.tail(2))
print('')
print(car_sales.loc[0])
print(car_sales.loc[3])
print(car_sales.iloc[3])
print(car_sales['Odometer (KM)'])
print(car_sales['Odometer (KM)'].mean())
print(car_sales[car_sales['Odometer (KM)']>10000])
print(car_sales.groupby(['Make']).mean())

print('')
#print(car_sales['Odometer (KM)'].plot())
#print(car_sales['Odometer (KM)'].hist())

# Convert Price (with $ sign) into Float
car_sales['Price'] = car_sales['Price'].replace('[\$,]', '', regex=True).astype(float)
print(car_sales)
car_sales['Price'] = car_sales['Price'].astype(int)
print(car_sales)
print(car_sales.dtypes)
car_sales['Make'] = car_sales['Make'].str.lower()
print(car_sales)    
print('')
# =============================================================================
# import csv file from a path - car_sales_missing data
# =============================================================================
print('<==== car_sales_missing ====>')
car_sales_missing = pd.read_csv('C:/Users/z011348/Desktop/ML/input/car-sales-missing-data.csv')
print(car_sales_missing)
print('')
car_sales_missing["Odometer"] = car_sales_missing["Odometer"].fillna(car_sales_missing["Odometer"].mean())
print(car_sales_missing)
print('')
car_sales_missing = car_sales_missing.dropna()
print(car_sales_missing)

car_sales_missing['seats'] = pd.Series([5,5,5,5])
print(car_sales_missing)
