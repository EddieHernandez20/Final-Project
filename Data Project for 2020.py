#!/usr/bin/env python
# coding: utf-8

# In[2]:


import itertools
import re

from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns


# In[3]:


raw2 = pd.read_csv('/Users/EduardoHernandez/Documents/result.csv')


# In[4]:


raw2.head()


# In[5]:


nRow, nCol = raw2.shape
print(f'There are {nRow} rows and {nCol} columns')


# In[6]:


# Distribution graphs (histogram/bar graph) of column data
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()


# In[7]:


# Correlation matrix
def plotCorrelationMatrix(df, graphWidth):
    filename = df.dataframeName
    df = df.dropna('columns') # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()


# In[8]:


# Scatter and density plots
def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number]) # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(df)
    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()


# In[9]:


raw2.head(50-100)


# In[10]:


raw2.dropna(axis=1, inplace = True)


# In[11]:


zero=[0]
raw2[raw2['|__positive'].isin(zero)]


# In[12]:


raw3=raw2.drop([573,574,760])


# In[13]:


pd.options.display.max_rows = 1000
raw3.head(990)


# In[14]:


def calc_rating(row):

    import math

    pos = row['|__positive']
    neg = row['|__negative']

    total_reviews = pos + neg

    average = pos / total_reviews

    score = average - (average*0.5) * 2**(-math.log10(total_reviews + 1))

    return score * 100


def get_unique(series):
    return set(list(itertools.chain(*series.apply(lambda x: [c for c in x.split(';')]))))

def process_revenue(df):
    df['est_revenue'] = df['|__owners'] * df['|__price']
    return df


def process_price(df):
    cut_points = [-1, 0, 4, 10, 30, 50, 1000]
    label_names = ['free', 'very cheap', 'cheap', 'moderate', 'expensive', 'very expensive']
    
    df['price_categories'] = pd.cut(df['|__price'], cut_points, labels=label_names)
    
    return df
#NOT WORKING
#def pre_process(df):
    # calculate ratings
    #df['total_ratings'] = df['|__positive'] + df['|__negative']
    #df['ratings_ratio'] = df['|__positive'] / df['total_ratings']
    #fdf['weighted_rating'] = df.apply(calc_rating, axis=1)
    # df = df.drop(['positive', 'negative'], axis=1)

df1 = pre_process(raw3)
#df1.dataframeName = 'result'

#nRow, nCol = df1.shape
#print(f'There are {nRow} rows and {nCol} columns after preprocessing')


# In[15]:


print(df1)


# In[16]:


raw3['total_reviews'] = raw3['|__positive'] + raw3['|__negative']
raw3['percent'] = raw3['|__positive'] / raw3['total_reviews']


# In[17]:


dfsample = raw3.sample(992)
dfsample.dataframeName = 'result'
plotPerColumnDistribution(dfsample, 10, 5)


# In[18]:


plotCorrelationMatrix(dfsample, 8)


# In[19]:


plotScatterMatrix(dfsample, 20, 10)


# In[20]:


dfsample.head()


# In[24]:


dfsample.to_csv(r'/Users/EduardoHernandez/Documents/dfsample20.csv')


# In[25]:


dfsample3 = dfsample


# In[27]:


dfsample3['Date'] = '2020'


# In[28]:


dfsample3.head()


# In[30]:


df20 = dfsample3 [['|__appid', '|__name', '|__positive', '|__negative', '|__owners', 
            '|__average_forever', '|__median_forever', '|__price', 'total_ratings', 
            'ratings_ratio','weighted_rating', 'Date']].rename(columns = {'|__appid':'appid', '|__name':'name', '|__positive':'positive_ratings', 
                                                                  '|__negative': 'negative_ratings', '|__owners': 'owners', '|__average_forever':'average_playtime',
                                                                  '|__median_forever':'median_playtime','|__price': 'price', 'Date':'Date'})


# In[31]:


df20


# In[32]:


df20.to_csv(r'/Users/EduardoHernandez/Documents/dfsample2020.csv')


# In[ ]:




