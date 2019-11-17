#%% [markdown]
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('classic')
dirpath = os.getcwd() 
path2add = 'E:\GWU FALL\Data Mining\PROJECT'
filepath = os.path.join( dirpath, path2add ,'data.csv')
fifa = pd.read_csv(filepath)  
fifa.head()

# %%
#To check whether index is unique and drop NA
print('\n',fifa.head(),'\n')
fifa.index
fifa.index.is_unique
dup=fifa.index.duplicated()
print(dup)
fifa_duo = fifa[dup]
fifa_duo.shape
fifa=fifa[dup == False]
fifa = fifa.reset_index()
fifa=fifa.dropna()
fifa.head()

# %%
