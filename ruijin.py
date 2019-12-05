#%% [markdown]
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
plt.style.use('classic')
dirpath = os.getcwd() 
filepath = os.path.join(dirpath ,'data.csv')
fifa = pd.read_csv(filepath)  
fifa.head()

# %%
#To check whether index is unique and drop NA
print('\n',fifa.head(),'\n')
fifa.index
print(fifa.index.is_unique)
fifa.columns
fifa = fifa.rename(columns={'Unnamed: 0': 'id', 'Club Logo':'Club_Logo', 'Preferred Foot':'Preferred_Foot', 'International Reputation':'International_Reputation', 'Weak Foot':'Weak_Foot', 'Skill Moves':'Skill_Moves', 'Work Rate':'Work_Rate', 'Body Type':'Body_Type', 'Real Face':'Real_Face', 'Jersey Number':'Jersey_Number', 'Loaned From':'Loaned_From', 'Contract Valid Until':'Contract_Valid_Until', 'Release Clause':'Release_Clause'})
fifa = fifa.drop(columns=['Loaned_From'])
fifa = fifa.dropna()

# %%
fifa['Wage'] = fifa['Wage'].map(lambda x: x[1:][:-1])
fifa['Wage'] = pd.to_numeric(fifa['Wage'])

# %%
plt.hist(fifa['Wage'], label='fifawage')
plt.show()
#From the plot we can see salaries are right skewed

# %%
#from the plot we can see high-level salaries are much less than the lower-level salary, and first we look into the low-level salary which is less than 200K and 50K  and 5K
#Salaries are right-skewed distributed in each group, and the frequencies become less and less as salaries go up
plt.hist(fifa[fifa['Wage']<200]['Wage'], label='fifawage',edgecolor='black', linewidth=1.2)

#%%
plt.hist(fifa[fifa['Wage']<50]['Wage'], label='fifawage',edgecolor='black', linewidth=1.2)

#%%
plt.hist(fifa[fifa['Wage']<5]['Wage'], label='fifawage',edgecolor='black', linewidth=1.2)

#%%
#Then we look into the high salaries, we can see the frequencies in each level are sparse and scattered, and also some high outlier exist
plt.hist(fifa[fifa['Wage']>200]['Wage'], label='fifawage',edgecolor='black', linewidth=1.2)

# %%
#BMI is an indicator of health condition, and we increase a new variable bmi(=height(m)/weight(kg)^2)
#fh=open(filepath, encoding="utf8")
#for aline in fh.readlines(): # readlines creates a list of elements; each element is a line in the txt file, with an ending line return character. 
#  h = int(aline.split(',').split("'")[0])+int(aline.split(',').split("'")[1])*0.1

#fifa['Weight']=0.453592*int(fifa['Weight'].strip('lbs'))

w2=[0.453592*int(w.strip('lbs')) for w in fifa['Weight']]
h2=[int(h.split("'")[0])+int(h.split("'")[1])*0.1 for h in fifa['Height']]
# for i in range(len(fifa['Weight'])):
fifa['Weight'] = fifa['Weight'].apply(lambda x: 0.453592*int(x.strip('lbs')))
fifa['Height'] = fifa['Height'].apply(lambda x: int(x.split("'")[0]) + int(x.split("'")[1])*0.1)
fifa['BMI'] = fifa['Weight'] / pow(fifa['Height']*30.48/100, 2)
# for m in range(len(w2)):
#     bmi=w2[m]/pow(h2[m]*30.48/100,2)
#     fifa['BMI'][m]=bmi

# fifa['BMI'].head()

# %%
plt.hist(fifa['Age'], label='fifaage',edgecolor='black', linewidth=1.2)
#From the age distribution we can see age 20 to 30 are the majority
#We expect there will be 'golden ages' for athletes, which means below or above the certain range of age, soccer players would not be at their time of best performance
#fifa['Age^2']=None
#age=[age for age in fifa['Age']]
#for n in list(range(len(age))):
#    agesquare=pow(age[n],2)
#    fifa['Age^2'][n]=agesquare
#fifa['Age^2'].head()
#We can have a look at plot of Wage & Age. The plot is normal distributed.
fig, axis = plt.subplots()
axis.plot(fifa.Age, fifa.Wage,color='g', linestyle="", marker="o", markersize=3)
plt.xlabel("Age")
plt.ylabel("Wage")
plt.title("Wage vs Age")
plt.show()

# %%
plt.hist(fifa['Jersey_Number'], label='fifanumber',edgecolor='black', linewidth=1.2)
#We can see Jersey Number under 40 are the majority

#%%
#Many fans and soccer believe there are 'lucky numbers'. And there is a rumor going like that numbers in a range tend to be picked as lucky numbers
#To test whether it makes sense, we got jursey number squared  
#fifa['Jersey_Number_Squared']=None
#number=[number for number in fifa['Jersey_Number']]
#for j in list(range(len(number))):
#    numbersquare=pow(number[j],2)
#    fifa['Jersey_Number_Squared'][j]=numbersquare
#fifa['Jersey_Number_Squared'].head()
fig, axis = plt.subplots()
axis.plot(fifa.Jersey_Number, fifa.Wage,color='g', linestyle="", marker="o", markersize=3)
plt.xlabel("Jersey Number")
plt.ylabel("Wage")
plt.title("Wage vs Jersey Number")
plt.show()
#From the plot we can see valuable players are accumulated around 10

#%%
plt.hist(fifa[fifa['Jersey_Number']<40]['Jersey_Number'], label='fifanumber',edgecolor='black', linewidth=1.2)
fig, axis = plt.subplots()
axis.plot(fifa[fifa['Jersey_Number']<40].Jersey_Number, fifa[fifa['Jersey_Number']<40].Wage,color='g', linestyle="", marker="o", markersize=3)
plt.xlabel("Jersey Number")
plt.ylabel("Wage")
plt.title("Wage vs Jersey Number")
plt.show()

# %%
#Then we build the liner model to see which variables could have significant effect on wages
modelwage = ols(formula='Wage ~ Age+Overall+Potential+Special+C(International_Reputation)+C(Weak_Foot)+C(Skill_Moves)+C(Work_Rate)+C(Body_Type)+C(Position)+Jersey_Number+BMI+Crossing+Finishing+HeadingAccuracy+ShortPassing+Volleys+Dribbling+Curve+FKAccuracy+LongPassing+BallControl+Acceleration+SprintSpeed+Agility+Reactions+Balance+ShotPower+Jumping+Stamina+Strength+LongShots+Aggression+Interceptions+Positioning+Vision+Penalties+Composure+Marking+StandingTackle+SlidingTackle+GKDiving+GKHandling+GKKicking+GKPositioning+GKReflexes', data=fifa).fit()
print(modelwage.summary())
#We drop the weak foot, body type, position, age, overall, special, BMI and skills

#%%
modelwage2 = ols(formula='Wage ~ Potential+Special+C(International_Reputation)+C(Skill_Moves)+C(Work_Rate)+Jersey_Number', data=fifa).fit()
print(modelwage2.summary())

