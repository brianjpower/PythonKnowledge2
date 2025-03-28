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

#%% md
#### Exercise 1
#Suppose I have a list of strings:

string_list = ['Python', 'is', 'fun']
print(" ".join(string_list))

#regex

import re


hem_string = 'He was an old man who  fished alone in a\n skiff in the  Gulf Stream and he had  gone eighty-four days \tnow without taking a fish.'
print(hem_string)

test = re.split('\s',hem_string)
print(test)
print(re.split('\s+',hem_string))

#Exercise 2

#%% md
#### Exercise 2
#Suppose I have a list of filenames and I want to extract all the file extensions (the values after the .)

files = ['lecture_9_code.py', 'graphs.jpeg', 'lecture_9.pdf', 'lecture9a.mp4']

#extensions = re.findall('\.\w+',str(files))
#print(x.strip('.') for x in extensions)
#print(extensions)

#print([re.findall(('\.\w+'),x) for x in files])

#%% md
#### Exercise 2


#Which of the following will split the filenames everywhere there's a full stop? (Select all that apply)
#print([re.findall('.',x) for x in files])
print([re.split('[\W]',x) for x in files])
#re.split('[.]',files)
#print([re.split('[.]',x) for x in files])
#re.split('[\W]',files)

#Exercise 6
#%% md
#### Exercise 6
#Write a piece of code to replace all instances of the word 'and' in
# the following string with '&'. Be sure to only replace 'and' when it
# appears as a word on its own and not as part of another word.
#
ex6 = "Amanda and Martin went to the beach and they built sand castles."

print(re.sub('(\sand\s)',' & ',ex6))


#############################################Section 8################################################
############################################Data Wrangling############################################

print("#########Section 8##############")

#####  Merging dataframes

#%% md
#### Exercise 1
#Suppose I want to merge the DataFrames below on both id and sex, but to keep all the IDs in the resulting DataFrame. Which is the best command?

#* `pd.merge(df_1,df_2)`
#* `pd.merge(df_1,df_2,on=['id'],how='outer')`
#* `pd.merge(df_1,df_2,on=['id','sex'],how='outer')`
#* `pd.merge(df_1,df_2,on=['id','sex'],how='right')`



df_1 = DataFrame({'id':range(134,139),'sex':['male','male','female','female','male'],'age':[36,21,53,33,29]})
df_2 = DataFrame({'id':range(135,139),'sex':['male','female','female','male'],'party':['FG','FF','SF','FF']})

df1_and_df2 = pd.merge(df_1,df_2, on=['id','sex'],how='outer')
print(df1_and_df2)
####### Other ways to combine dataframes
#Can also stack datasets one on top of the other or join them side by side rather than merging
summer_dict = {'Year': [2015, 2016, 2017, 2018, 2019, 2020] ,'mean_temp': [19,17,20,16,17,18],
    'max_temp': [25,20,27,21,22,25],'min_temp': [10,12,8,9,10,11]}
summer_df = DataFrame(summer_dict,index=summer_dict['Year'])
winter_dict = {'Year': [2012, 2013, 2014,2015,2016],'mean_temp': [13,12,15,10,9],
    'max_temp': [18,16,20,14,14],'min_temp': [2,0,-1,0,-2],'mean_rain': [2,4,5,4,3]}
winter_df = DataFrame(winter_dict,index=winter_dict['Year'])
print(summer_df,'\n')
print(winter_df,'\n')

summer_winter_stack_df = pd.concat([summer_df,winter_df],sort=True)
print(summer_winter_stack_df)

summer_winter_join = summer_df.join(winter_df,lsuffix='_winter',rsuffix='_summer')
print(summer_winter_join)


#Excercise 2

#%% md

##### Duplicates

#Exercise 3
#%% md
#### Exercise 3
#Load in the diamonds dataset from Week 5 and determine how many duplicate rows there are.
#print(diamonds_df)


print(diamonds_df.duplicated().sum())
##### Mapping and replacing

#my_map = {1:'Under 18',18:'18-24',25:'25-34',35:'35-44',45:'45-49',50:'50-55',56:'56+'}
#users['age_grp'] = users['age'].map(my_map)
#print(users.head())

path = 'C:/Users/brian/OneDrive/Documents/MSC Data Analytics/Data Prog with Python/Week-8-data/'
ratings = pd.read_csv(path + 'ratings.dat',sep='::',engine='python',names=['user_id','movie_id','rating','timestamp'])
#path='C:/Users/brian/OneDrive/Documents/MSC Data Analytics/Data Prog with Python/Week-8-data/'
#ratings = pd.read_csv(path+'ratings.dat',sep='::',names=['user_id','movie_id','rating','timestamp'],engine='python')

print(ratings.head())

ratings_map = {1:'Really Bad',2:'Bad',3:'Average',4:'Good',5:'Very Good'}

#ratings['critique'] = ratings['rating'].map(ratings_map)
#print(ratings)

ratings['rating_1'] = ratings['rating'].map(ratings_map)
print(ratings)

ratings['rating_2'] = ratings['rating'].map(lambda x: ratings_map[x])
print(ratings)


##### Cutting Values

npr.seed(111)
ages = Series(npr.poisson(lam=30,size=100))
bins = [0,20,40,60]
age_groups = pd.cut(ages,bins)
#print(age_groups)

#age_groups_4 = pd.cut(ages,4)
#print(pd.value_counts(age_groups_4))

npr.seed(27)
x = Series(npr.randint(0,100,1000))

print(x)


#print(movie_cuts)

"""
import pandas as pd

df = pd.DataFrame({'Age': [45, 60, 15, 30], 'Diabetes': [0, 1, 0, 1]})
print(df)
#print(df.loc[df['Age'] > 40]['Diabetes'].mean())
print(df[df['Age']>40]['Diabetes'].mean())
print(df.loc[df['Age']>40])


import pandas as pd

df = pd.DataFrame({
'Weight': [45, 88, 56, 15, 71],
'Name': ['Sam', 'Andrea', 'Alex', 'Robin', 'Kia'],
'Age': [14, 25, 55, 8, 21]
}, index=['Row_1', 'Row_2', 'Row_3', 'Row_4', 'Row_5'])

result = df.loc['Row_2', 'Name']
print(result) # Output: Andrea
print(df)
result2 = df.loc[:,['Weight', 'Name']]
print(result2)

print(df.loc[df['Age'] < 20]['Weight'].mean())
print(df.loc[df['Age'] > 20]['Weight'].mean())
#print(result)
"""

##### Dummy Values

bins = [0,20,40,60,80,100]
int_cut = pd.cut(x,bins)
print(int_cut.value_counts())
age_30 = pd.cut(ages,[0,30,100],labels=['Under 30','over 30'])
print(pd.get_dummies(age_30).head(10))
#%% md
#### Exercise 6
#Use a combination of the `cut` and `get_dummies` commands to create set of dummy variables
# *Bad* and *Good* for the movie ratings data.
# The dummy variable *Bad* should be 1 if the rating is 2 or less and 0 otherwise,
# while the dummy variable *Good* should be 1 if the rating is 3 or more and 0 otherwise.
movie_cuts = pd.cut(ratings['rating'],bins=[0,2,5],labels=['bad','good'])
print(pd.get_dummies(movie_cuts).tail(20))
print(ratings.tail(20))

##### Standardising Data

#To standardise a dataset we refactor it such that it isn't a raw magnitude but rather represented as the number of
#standard deviations form the mean.  To do this, we take the dataset, subtract all the means from the raw data
#and the divide this by that columns standard deviation

print(summer_df)
print(summer_df.describe())
summer_df_standard = (summer_df[['mean_temp','max_temp','min_temp']] - summer_df[['mean_temp','max_temp','min_temp']].mean() ) / summer_df[['mean_temp','max_temp','min_temp']].std()
print(summer_df_standard)
#print((summer_df[['mean_temp','max_temp','min_temp']] - summer_df[['mean_temp','max_temp','min_temp']].mean())/summer_df[['mean_temp','max_temp','min_temp']].std())


##%% md
#A new medical test is in development for detecting Ebola that is much faster,
# though less accurate, than that currently available. The new test has been applied to
# some patients who are known (from the old test) to have the disease or not.
# The results of these patients are available in `ebola_test.csv`.
# The first column of this data set (called `prob`) is the probabiliy under the new test that the
# patient has Ebola. The second column (`ebola`) is the result from the older test which definitively
# says whether the  patient has ebola (`ebola = 1`) or not (`ebola=0`).

#Begin by loading in the data and having a look at it.

ebola_test = pd.read_csv(path+'ebola_test.csv')
print(ebola_test.head(20))
print(ebola_test.tail(20))
num_infected = ebola_test[ebola_test['ebola']==1].count()[0]
print(num_infected)
perc_infected = (num_infected/ebola_test.count()[0])*100
print(f"The percentage of patients infected is {perc_infected.round(2)}%")


#Now let's make a cutoff of the prob data and use this to predict infection.  Let's choose a cut-off of
#0.15 and calculcate the false postive and false negative rates.  Try to tune the cut-off then with ROC?

my_cutoff = [0,0.15,1]
ebola_test['pred'] = pd.cut(ebola_test['prob'],my_cutoff,labels=[0,1])
print(ebola_test)

false_positive = ((ebola_test[(ebola_test['pred']==1) & (ebola_test['ebola']==0)].count()[0]) / (ebola_test[(ebola_test['pred']==1) & (ebola_test['ebola']==0)].count()[0] + ebola_test[ebola_test['ebola']==0].count()[0]))*100
print(f"The false_positive rate is {false_positive}%")
false_negative = ((ebola_test[(ebola_test['pred']==0) & (ebola_test['ebola']==1)].count()[0]) / (ebola_test[(ebola_test['pred']==0) & (ebola_test['ebola']==1)].count()[0] + ebola_test[ebola_test['ebola']==1].count()[0]))*100
print(f"The false negative rate is {false_negative}%")

def FPR(cutoff):
    my_cutoff = [0, cutoff, 1]
    ebola_test['pred'] = pd.cut(ebola_test['prob'], my_cutoff, labels=[0, 1])
    return ((ebola_test[(ebola_test['pred']==1) & (ebola_test['ebola']==0)].count()[0]) / (ebola_test[(ebola_test['pred']==1) & (ebola_test['ebola']==0)].count()[0] + ebola_test[ebola_test['ebola']==0].count()[0]))*100

print(FPR(0.10))


def FPR_new(dataframe, cutoff, col):
    my_cutoff = [0, cutoff, 1]
    dataframe['pred'] = pd.cut(dataframe[col], my_cutoff, labels=[0, 1])
    return ((dataframe[(dataframe['pred']==1) & (dataframe['ebola']==0)].count()[0]) / (dataframe[(dataframe['pred']==1) & (dataframe['ebola']==0)].count()[0] + dataframe[dataframe['ebola']==0].count()[0]))*100

print(FPR_new(ebola_test,0.17,'prob'))

def FNR(dataframe, cutoff, col):
    my_cutoff = [0, cutoff, 1]
    dataframe['pred'] = pd.cut(dataframe[col], my_cutoff, labels=[0, 1])
    return ((dataframe[(dataframe['pred']==0) & (dataframe['ebola']==1)].count()[0]) / (dataframe[(dataframe['pred']==0) & (dataframe['ebola']==1)].count()[0] + dataframe[dataframe['ebola']==1].count()[0]))*100

print(FNR(ebola_test,0.15,'prob'))



def find_min_fpr(list_of_cuts):
    list_of_fpr = []
    min_fpr = 100
    min_cut = 0
    for i in list_of_cuts:
        list_of_fpr.append(FPR_new(ebola_test,i,'prob'))
        if FPR_new(ebola_test,i,'prob') < min_fpr:
            min_fpr = FPR_new(ebola_test,i,'prob')
            min_cut = i
    min_fpr = min(list_of_fpr)
    return min_fpr, min_cut
    #return list_of_fpr

def find_min_fnr(list_of_cuts):
    list_of_fnr = []
    min_fnr = 100
    min_cut = 0
    for i in list_of_cuts:
        list_of_fnr.append(FNR(ebola_test,i,'prob'))
        if FNR(ebola_test,i,'prob') < min_fnr:
            min_fnr = FNR(ebola_test,i,'prob')
            min_cut = i
    min_fnr = min(list_of_fnr)
    return min_fnr, min_cut
"""
print(np.array(list(np.arange(0.15,0.3,0.001))))
print(find_min_fpr(list(np.arange(0.15,0.17,0.001))))
print(FPR_new(ebola_test,0.17,'prob'))
print(find_min_fnr(list(np.arange(0.1,0.3,0.001))))
print(FNR(ebola_test,0.1,'prob'))

"""
#/ (ebola_test[ebola_test['pred']==1].count()[0] + ebola_test[ebola_test['pred']==0].count()[0])
#false_negative = ebola_test[ebola_test['pred']==0].count()[0] / (ebola_test[ebola_test['pred']==1].count()[0] + ebola_test[ebola_test['pred']==0].count()[0])
#print(f"The false positive rate is {false_positive*100}%")
#print(f"The false negative rate is {false_negative*100}%")

#%% md
#A related concept to the choice of cut-offs is the Receiver Operator Characteristic (ROC) curve.
#The ROC curve aims to quantify how well a classifier beats a random classifier for any level of
#probability cut-off. An introduction can be found here: http://en.wikipedia.org/wiki/Receiver_operating_characteristic
#The idea is to plot the false positive rate against the true positive rate for every cut-off value.

#Your final task is to write a piece of code to compute the ROC curve.
#Your ROC curve code should perform the following steps

#1. Find the unique values in the probability column
#2. Use each of these unique values as a cutoff value and <br>
#a. Classify all the obs as either positive or negative based on the current cutoff value<br>
#b. Calculate the false positive rate (FPR) and true positive rate (TPR)<br>
#3. Plot the false positive rate versus true positive rate.

#**Note 1:** FPR and TPR must be arrays/Series of the same length as the array/Series of unique values<br>
#**Note 2:** Be careful that you calculate FPR and FNR correctly when at the largest unique probability value<br>

list_of_cuts = np.sort(ebola_test['prob'].unique())
print(list_of_cuts)
fpr = []
tpr = []

for cutoff in list_of_cuts:
    ebola_test['pred'] = pd.cut(ebola_test['prob'], [0, cutoff, 1], labels=[0, 1])
    true_pos = ebola_test[(ebola_test['pred']==1) & ebola_test['ebola']==1].count()[0]
    true_neg = ebola_test[(ebola_test['pred']==0) & ebola_test['ebola']==0].count()[0]
    false_pos = ebola_test[(ebola_test['pred']==1) & ebola_test['ebola']==0].count()[0]
    false_neg = ebola_test[(ebola_test['pred']==0) & ebola_test['ebola']==1].count()[0]

    fpr.append(false_pos/(false_pos + true_neg))
    tpr.append(true_pos/(true_pos + false_neg))

 # Plot the ROC curve
print(fpr)
print(tpr)
print(list_of_cuts)
import matplotlib.pyplot as plt
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.show()



#print(fpr)







