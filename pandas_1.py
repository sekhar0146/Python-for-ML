import pandas as pd
#import sys
#sys.path.append('/usr/local/lib/python2.7/site-packages')

car = pd.Series(["Duster", "Kwid","Triber"])

colours = pd.Series(["white", "Red", "Blue"])

car_data = pd.DataFrame(
    {
     "Car_model": car,
     "Car_colour": colours
     }
    )
print(car_data)
print(' ')

# =============================================================================
# import csv file from a path
# =============================================================================
emp_data = pd.read_csv('C:/Users/z011348/Desktop/ML/input/emp.csv')
print(emp_data)
emp_data.to_csv('C:/Users/z011348/Desktop/ML/output/emp_save.csv', index=False)
print(' ')
# =============================================================================
# import csv file from web/url
# =============================================================================
# =============================================================================
# heart_disease = pd.read_csv("https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/heart-disease.csv")
# print(heart_disease)
# =============================================================================
print(emp_data.dtypes)  # dataframe data types
print(emp_data.columns)  # dataframe columns in LIST format
print(emp_data.index) # indexing 
print(emp_data.describe())
print(emp_data.info())
print(len(emp_data))
print(emp_data.sum())
print(emp_data.mean())
print(emp_data.head()) # top 5 rows
print(emp_data.head(3))
print(emp_data.tail(3)) # display last recs
print(' ')

nums = pd.Series(["one", "two", "three"])
print(nums)
nums1 = pd.Series(["one", "two", "three"], index=[3,2,1]) # change the index
print(nums1)
print('')
print(nums1.loc[3])
print('')
print(emp_data.loc[5])
print('')
# loc - index; iloc - position 
print(emp_data.ename)
print(emp_data["eno"])  # select single col
print(emp_data[['eno', 'ename', 'esal']]) # select multiple cols
print(emp_data[emp_data["ename"]=="abx"]) 

print('')

# =============================================================================
# plot
# =============================================================================
#print(emp_data["esal"].plot())

# =============================================================================
# histogram
# =============================================================================
#print(emp_data["esal"].hist())

print(emp_data)
print(emp_data["ename"].str.upper())
emp_data["ename"] = emp_data["ename"].str.upper()
print(emp_data)

print("")
emp_data["eloc"] = emp_data["eloc"].fillna(" ")
print(emp_data)

print("")
emp_data_dropped_missing = emp_data.dropna()
print(emp_data_dropped_missing)

# add a new column
print("")
emp_data["revsal"] = pd.Series([1500.0, 3200.0, 4500.0])
print(emp_data)
emp_data["revsal"] = emp_data["revsal"].fillna(0)
print(emp_data)

# drop column
print('')
emp_data = emp_data.drop("revsal", axis=1)
print(emp_data)

# add a new column
print("")
emp_data["revsal"] = emp_data["esal"]*(5/100)+emp_data["esal"]
print(emp_data)

# sample(frac) : FRAC=0.5 = 50% DATA, 1 = 100% DATA
print('')
emp_data_shuffled = emp_data.sample(frac=1)
print(emp_data_shuffled)

#reset the indexs
print('')
emp_data_shuffled.reset_index(drop=True, inplace=True)
print(emp_data_shuffled)

# Manipulations/convert on any columns with lambda 
print('')
emp_data['PFamount'] = emp_data['esal'].apply(lambda a: (12/100)*a)
print(emp_data)
print('')
