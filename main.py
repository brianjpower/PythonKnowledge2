# This is a sample Python script.
from numpy.random import normal

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    from pandas import DataFrame,Series

#This is a dev branch test

population = {'Year':range(2010,2016),
    'Births':[75174,74033,71674,68954,67295,65536],
    'Deaths':[27961,28456,29186,29504,29252,30127]}
population_df = DataFrame(population,index=population["Year"],columns=["Births","Deaths"])
print(population_df)

#The index is immutanble buy the index can be extended with the reindex method

population_df2 = population_df.reindex(range(2010,2020))
print(population_df2)

#If we don't want NaN filled in for data in reindexed rows, can assign fill-in values

population_df3 = population_df.reindex(range(2010,2020),fill_value=60000)
print(population_df3)

#of extrapolate with forward fill
population_df4 = population_df.reindex(range(2010,2020),method="ffill")
print(population_df4)

#Can also use reindex to add columns in a similar fashion
population_df5 = population_df.reindex(columns=["Births","Deaths","Difference"])
population_df5["Difference"] = population_df5["Births"] - population_df5["Deaths"]
print(population_df5)


growth = DataFrame({'Height':[123,135,139,142,155],'Weight':[20,25,28,29,41]},index=[2011,2013,2014,2015,2017])
print(growth)

growth_new = growth.reindex(range(2011,2018),method="bfill")
print(growth_new)

#indexes combine when adding or subtracting series or dataframes
height_2019 = Series([67,98,72,82],index=['Alan','Bill','Charles','David'])
height_2020 = Series([112,78,90],index=['Bill','Alan','David'])
print(height_2020-height_2019)

kids_2020 = DataFrame({'Height':[112,78,90,82],'Weight':[25,13,16,14],'Age':[6,2,4,3]},
                      index=['Alan','Bill','Charles','David'])
kids_2019 = DataFrame({'Height':[67,98,72],'Weight':[10,19,11]},index=['Bill','Alan','David'])
print(kids_2020-kids_2019)


#%% md
#### Exercise 2
#Add the DataFrames below to find the total number of apples and oranges
# collected across the two farms each month. If there is no entry for a particular month,
# you can assume that the farm collected no fruit that month. With this in mind,
# how should you deal the missing values?
#
farm1 = DataFrame({'apples':[12,27,36,50,103,165,152,202,163,177],'oranges':[45,13,105,67,178,154,204,187,186,73]},
                        index=['Jan','Feb','Mar','Apr','May','Jul','Aug','Sep','Oct','Nov'])
farm2 = DataFrame({'apples':[34,10,6,68,167,208,199,164,65],'oranges':[2,13,5,45,61,52,82,64,17]},
                        index=['Jan','Feb','May','Jun','Jul','Aug','Sep','Nov','Dec'])
#print(farm1.add(farm2,fill_value=0))
month_order = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
print(farm1.add(farm2,fill_value=0).reindex(month_order))

#Sorting and ranking
from pandas import DataFrame,Series
names = Series([4,6,2,7,1],index=['Vincent','William','John','Gary','Peter'])
print(names.sort_index(ascending=False))
print(names.rank())


print(names.sort_values(ascending=True))

print(population_df.sort_values(by='Deaths'))

employees_df = DataFrame({'salary_eur': [71652,31223,21156,54633,21156],'grade': [5, 2, 1, 4, 1],
                          'year_service': [7, 4, 10, 3, 6]},index=['Vincent','William','John','Gary','Peter'])


print(employees_df.sort_values(by=['grade','year_service'], ascending =[True,False]))

#Computing summary stats

import numpy.random as npr
npr.seed(101)
rand_df = DataFrame({'normal':npr.randn(20),
                     'binomial':npr.binomial(10,0.2,20),
                     'poisson':npr.poisson(1,20),
                     'colours':npr.choice(['red','blue','green'],size=20)})
print(rand_df.head())
print(rand_df.describe())
print(rand_df[['normal','binomial','poisson']].mean(axis=0))
#or writh indices
print(rand_df.iloc[:,:3].mean(axis=0))

#or by extracting numeric columns
numeric_columns = rand_df.select_dtypes(include='number')  # Select only numeric columns
print(numeric_columns.mean(axis=0))

#or drop the non-numeric column
print(rand_df.drop('colours',axis=1).mean(axis=0))

#correlations

#%% md


#To compute the *correlation* or *covariance* we must again
# drop the categorical variables to apply `corr` and `cov`, respectively.

print(rand_df.drop('colours',axis=1).corr())
print(rand_df.drop('colours',axis=1).cov())

#individual correlations can also be found
print(rand_df.normal.corr(rand_df.binomial))
#Similarly, corrwith can be used to findall correlations with one variable / column.

print(rand_df.colours.unique())

print(rand_df.colours.value_counts())

#%% md
#### Exercise 4
#Load in the `diamonds.csv` dataset, and use it to answer the following questions:
#1. What is the average price of a diamond in this dataset?
#2. What is the variance of the table measurement?
#3. How many unique diamond colours are there?
#4. What is the most common cut?
#
path = 'C:/Users/brian/OneDrive/Documents/MSC Data Analytics/Data Prog with Python/Full course/'
#diamonds = open(path + 'diamonds.csv').read()
diamonds_df = pd.read_csv(path+'diamonds.csv')
print(diamonds_df.head())
#Print average price of diamonds
print(diamonds_df.price.mean())
#Variane of the table measurement
print(diamonds_df.table.var())

print(diamonds_df.color.describe())
print(diamonds_df.color.unique())
print(diamonds_df.cut.value_counts())


#%% md
#### Exercise 5
#The DataFrame below contains the component grades for 5 students on a
# particular module. It was agreed that students who missed the midterm exam would receive
# the average grade of the other students, while students who did not submit the continuous assessment
# component or did not sit the final exam would receive zero for those components.
# Use the `fillna` method to update the students grades accordingly.

grades = DataFrame({'continuous_assessment':[64,78,90,None,58],'midterm':[52,None,83,57,28],
                    'exam':[None,72,78,61,None]},index=['Sarah','Michael','Robert','Ellen','Kevin'])
grades.midterm = grades.midterm.fillna(grades.continuous_assessment.mean())
grades.exam = grades.exam.fillna(0)
grades.continuous_assessment = grades.continuous_assessment.fillna(0)
#print(grades.fillna(grades.mean()))
print(grades)

salary_df = DataFrame({'salary':[53215,112454,np.nan,30493,None],
                    'grade':[5,None,2,np.nan,9]},
                     index=['Margaret','Stephen','Joe','Matthew','Nelson'])
salary_df.salary = salary_df.salary.fillna(salary_df.salary.mean())
print(salary_df)
#salary_df.grade = salary_df.grade.fillna(salary_df.grade.mean())
salary_df.grade = salary_df.grade.fillna(4)
#salary_df.grade = salary_df.grade.fillna(0)
#salary_df.grade = salary_df.grade.fillna(0)
print(salary_df)

def f(x):
    return x.max() - x.min()

print(grades.apply(f))
print(salary_df.apply(f))
print(salary_df.apply(f,axis=1))

def f2(x):
    return Series([x.max(), x.min()],index=['max','min'])
print(salary_df.apply(f2))

g = lambda x: x.max() - x.min()

print(salary_df.apply(g))
print(salary_df.apply(g,axis=1))

def final_grade(x):
    return x[0]*0.2 + x[1]*0.1 + x[2]*0.7
final_grades = Series(grades.apply(final_grade,axis=1)).sort_values(ascending=False)
print(grades.apply(final_grade,axis=1))
print(final_grades)
print(final_grades.iloc[0])

#Additional Exercises
weather = pd.read_csv(path+'weather.csv',index_col='date')
weather_orig = weather.copy()
print(weather.head())
print(weather.describe())
weather.mean_temp = weather.mean_temp.fillna(weather.mean_temp.mean())
weather.max_temp = weather.max_temp.fillna(weather.max_temp.mean())
weather.min_temp = weather.min_temp.fillna(weather.min_temp.mean())
weather.rain=weather.rain.fillna(weather.rain.mean())
print(weather.describe())
print(weather_orig.describe())
print((weather.describe() - weather_orig.describe()).round(2))
##%% md
#Using the `sort_values` method, or otherwise, find the months with the most rain. Give the month and year of the 3rd wettest month in this dataset.
print(weather.rain.sort_values(ascending=False).head(3))
def sub_means(x):
    return x - x.mean()

print(weather.apply(sub_means))

weather_deltas = weather.apply(sub_means)

print(weather_deltas.describe())



#############################################Section 7################################################
######################String manip and reg expressions################################################

print("####Section 7")

