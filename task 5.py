#!/usr/bin/env python
# coding: utf-8

# In[8]:


#Step 1: Importing required libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


#checking the location of the repository

import os
print(os.getcwd())


# In[32]:


#Step 2:loading the dataset

df = pd.read_csv("train.csv", encoding='ISO-8859-1')
df


# In[33]:


#Step 3:Standardize column names (lowercase and replace spaces with underscores)

df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")


# In[34]:


# Step 4: View basic info

print("Initial dataset shape:", df.shape)
print("\nColumn-wise missing values:\n", df.isnull().sum())


# In[35]:


#Step 5:removing duplicates

df = df.drop_duplicates()
print("\nShape after removing duplicates:", df.shape)


# In[36]:


#Step 6:Fill missing Age with median or mean

df['age'].fillna(df['age'].median(), inplace=True)
df.isnull().sum()


# In[37]:


#Step 7: Fill missing Embarked with the most frequent value

df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
df.isnull().sum()


# In[38]:


#Step 8: relacing missing values of cabin with 'unknown'

df['cabin'] = df['cabin'].fillna("unknown")
df.isnull().sum()


# In[40]:


# Step 10: Export the cleaned dataset to a new CSV file


df.to_csv("train_cleaned.csv", index=False)
print("\nCleaned dataset saved as 'train_cleaned.csv'")


# In[ ]:





# In[ ]:





# In[ ]:





# In[4]:


#Step 11:loading the cleaned dataset

df1 = pd.read_csv("train_cleaned.csv", encoding='ISO-8859-1')
df1


# In[ ]:





# In[11]:


#Step 12:Standardize column names (lowercase and replace spaces with underscores)

df1.columns = df1.columns.str.strip().str.lower().str.replace(" ", "_")


# In[ ]:





# In[12]:


df1.head()


# In[ ]:





# In[13]:


df1.describe()


# In[ ]:





# In[23]:


df1.mode()


# In[ ]:





# In[14]:


df1.info()


# In[ ]:





# In[ ]:





# In[16]:


#converting the columns Sex and Title from objects to category

df1['sex'] = df1['sex'].astype('category')
df1['embarked'] = df1['embarked'].astype('category')


# In[ ]:





# In[ ]:





# # Univariate Analysis

# In[19]:


# Numerical: Histograms


sns.histplot(df1['age'], kde=True)


# In[20]:


sns.histplot(df1['fare'], kde=True)


# since mean fare is 32, most bars are concentrated before 100 and min fare is 100 and mode of fare is 8 hence the highest bar is near 0

# In[ ]:





# In[26]:


#Categorical: Countplots

sns.countplot(x='survived', data=df1)


# In[27]:


sns.countplot(x='pclass', data=df1)


# In[28]:


sns.countplot(x='sex', data=df1)


# In[29]:


sns.countplot(x='embarked', data=df1)


# In[37]:


sns.countplot(x='sex', hue='survived', data=df1)


# In[ ]:





# In[ ]:





# # Bivariate Analysis

# In[ ]:





# In[39]:


# Categorical vs Target (Survived)

sns.barplot(x='sex', y='survived', data=df1)


# y='Survived' is binary (0 = died, 1 = survived)
# 
# So barplot calculates average of 0s and 1s
# 
# This average = percentage of people who survived
# 
# Females had a 70% survival rate
# Males had a 15% survival rate

# In[34]:


sns.barplot(x='survived', y='age', data=df1)


# The average age of people who did not survive is between 29 and 30 and the average age of people who survived is between 28 and 29

# In[31]:


sns.barplot(x='pclass', y='survived', data=df1)


# The passengers from class 1 had 63% survival rate, those from class 2 had 48% survival rate and those from class 3 had 24% survival rate

# In[ ]:





# In[32]:


# Numerical vs Target

sns.boxplot(x='survived', y='age', data=df1)


# Non-survivors (0): Median around 28–29
# Survivors (1): Median around 25–26
# Survivors tended to be slightly younger on average
# 
# Both groups have outliers, especially above age 60
# Some elderly people in both groups, but more elderly non-survivors
# 
# Minimum age is close to 0 in both groups
# Max age for survivors is slightly higher (80) than for non-survivors (70+)
# 
# 
# *  Younger people were slightly more likely to survive
# * There’s a wide age distribution in both groups, but survivors lean younger
# * Elderly passengers were more likely not to survive
# * Outliers (elderly survivors) exist but are rare as compared to elderly non-survivors

# In[41]:


sns.violinplot(x='survived', y='fare', data=df1)


# For Survived = 0 (Did not survive):
# Most passengers paid low fares (clustered near the bottom).
# The distribution is tightly packed, with a few outliers who paid higher fares.
# There’s less variation in fare amounts among those who didn’t survive.
# 
# 
# 
# For Survived = 1 (Survived):
# There's greater spread in fare values.
# While many survivors also paid low fares, there is a noticeable tail extending upward, indicating some survivors paid very high fares (possibly from 1st class).
# This group has more variance and outliers, showing a wider range of fares.

# In[ ]:





# In[ ]:





# # Multivariate Analysis

# Heatmap (Correlation)

# In[42]:


plt.figure(figsize=(10,6))
sns.heatmap(df1.corr(), annot=True, cmap='coolwarm')


# Fare and Pclass are important features that influenced survival.
# Age, SibSp, and Parch have weaker influence individually, but may matter when combined (feature interactions).
# This heatmap backs up your visualizations and can help guide feature selection or modeling choices if you're building a prediction model.

# In[ ]:





# Pairplot

# In[44]:


sns.pairplot(df1[['survived', 'pclass', 'age', 'fare']], hue='survived')


# Pclass:
# Blue (did not survive) is concentrated at class 3.
# Orange (survived) is more spread, with peaks in class 1 and 2 — higher class → better survival chances.

# Age:
# Both survived and not survived are centered around 20–30.
# Slightly more orange at younger ages — younger passengers had slightly higher chances.

# Fare:
# Survivors (orange) show a longer tail toward high fares — again showing that higher fares = better survival odds.

# ➤Pclass vs Age:
# Clear vertical clustering (1, 2, 3) due to Pclass being categorical.
# More orange in 1st class, even among older passengers.
# 
# ➤ Pclass vs Fare:
# Strong separation: 1st class fares are significantly higher.
# Most high-fare outliers are survivors.
# 
# ➤ Age vs Fare:
# No strong linear relationship.
# Survivors spread across all ages and fares, but those with very high fares mostly survived.

# Pclass and Fare show a strong relationship with survival.
# 
# Age has some influence (especially for very young), but not as strong.
# 
# There’s multicollinearity between Pclass and Fare, so in modeling, you might consider using just one or applying dimensionality reduction.

# In[ ]:





# In[ ]:





# # Survival Rate Analysis

# In[46]:


survival_rate = df1.groupby('sex')['survived'].mean()
print(survival_rate)


# In[ ]:





# In[ ]:





# # Class-wise Age Distribution

# In[47]:


sns.violinplot(x='pclass', y='age', hue='survived', data=df1, split=True)


# Pclass 1 (left):
# * Ages are spread widely, especially for non-survivors (blue).
# * Slightly more survivors (orange) around the 30–50 age range.
# * Survival and death occurred across all ages — survival not sharply age-dependent here.
# 
# 
# Pclass 2 (middle):
# * More balance between survival and non-survival.
# * Survivors slightly cluster around age 20–40.
# * Younger adults in 2nd class had moderate survival chances.
# 
# Pclass 3 (right):
# * The largest class, with more non-survivors (blue).
# * Survivors (orange) are more concentrated in younger age groups, especially kids and young adults.
# * Suggests young 3rd class passengers had higher chances of survival than older ones.

# Class matters — 1st class had better chances across all ages.
# 
# Younger passengers in 3rd class were more likely to survive than older ones.
# 
# Age and survival correlation depends heavily on Pclass — strong interaction effect.

# In[ ]:





# In[ ]:





# # Compare Survived vs Not-Survived Subsets

# In[51]:


df1[df1['survived']==1].describe()
df1[df1['survived']==0].describe()


# In[ ]:




