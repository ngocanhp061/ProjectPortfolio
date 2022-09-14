#!/usr/bin/env python
# coding: utf-8

# # Project: Titanic - Machine Learning from Disaster
# 

# In[1]:


import pandas as pd
import numpy as np 


# In[2]:


train_df = pd.read_csv("titanic_train.csv")
test_df = pd.read_csv("titanic_test.csv", index_col="PassengerId")


# In[3]:


train_df.columns


# In[4]:


test_df.columns


# In[5]:


#preview data
train_df.head()


# In[6]:


train_df.set_index(train_df.PassengerId, inplace=True)


# In[7]:


train_df.head()


# In[8]:


train_df.drop('PassengerId', axis=1, inplace=True)


# In[9]:


train_df


# In[10]:


test_df.head()


# # 1. Feature Classification

# In[11]:


train_df.info()


# In[12]:


test_df.info()


# - Categorical:  Survived, Sex, Embarked, Pclass(ordinal), SibSp, Parch
# - Numerical: (continuous) Age, Fare (discrete)
# - Mix types of data: Ticket, Cabin 
# - Contain Error/Typo: Name
# - Blank or Null: Cabin, Age, Embarked
# - Various Data Type: string, int, float

# In[13]:


train_df["Survived"] = train_df["Survived"].astype("category")


# In[14]:


train_df["Survived"].dtype


# In[15]:


features = ["Pclass", "Sex", "SibSp", "Parch", "Embarked"]
def convert_cat(df, features):
    for feature in features:
        df[feature] = df[feature].astype("category")
convert_cat(train_df, features)
convert_cat(test_df, features)


# In[16]:


train_df.info()


# ### Distribution of Numerical feature values across the samples

# In[17]:


train_df.describe()


# ### Distribution of Categorical feature values across the samples

# In[18]:


train_df.describe(include=['category'])


# In[ ]:





# # 2. Exploratory Data Analysis
# ### 2.1.Correlating categorical features
# - Categorical:  Survived, Sex, Embarked, Pclass(ordinal), SibSp, Parch

# #### Target Variable: `Survived`

# In[19]:


train_df["Survived"].value_counts().to_frame()


# In[20]:


train_df["Survived"].value_counts(normalize=True).to_frame()


# Only 38% survived in the disaster.So the training data suffers from data imbalance but it is not severe which is. That's why I will not consider techniques like sampling to tackle the imbalance
# #### `Sex`

# In[22]:


train_df["Sex"].value_counts(normalize=True).to_frame()


# In[23]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[26]:


sns.countplot(data=train_df, x='Sex', hue='Survived', palette='Blues')


# - Remaining Categorical Feature Columns

# In[29]:


cols = ['Sex', 'Embarked', 'Pclass', 'SibSp', 'Parch']
n_rows = 2
n_cols = 3
fig, ax = plt.subplots(n_rows, n_cols, figsize=(n_cols*3.5, n_rows*3.5))
for r in range(0, n_rows):
    for c in range(0, n_cols):
        i = r*n_cols + c #index to loop through list "cols"
        if i < len(cols):
            ax_i = ax[r,c]
            sns.countplot(data=train_df, x=cols[i], hue='Survived', palette='Blues', ax=ax_i)
            ax_i.set_title(f"Figure{i+1}: Survival Rate vs {cols[i]}")
            ax_i.legend(title='', loc='upper right', labels=['Not Survived', 'Survived'])
ax.flat[-1].set_visible(False) #Remove the last subplot
plt.tight_layout()


# #### Observation:
# - Survival Rate:
#     - Fig 1: Female survival rate > male
#     - Fig 2: Most people embarked on Southampton, and also had the highest people not survived
#     - Fig 3: 1st class higher survival rate
#     - Fig 4: People going with 0 `SibSp` are mostly not survived. The number of passenger with 1-2 family members has a better chance of survival
#     - Fig 5: People going with 0 `Parch` are mostly not survived
#     
# ### 2.2. EDA for Numerical Features
# - Numerical Features: `Age`(continuous), `Fare`(discrete)

# #### Age

# In[31]:


sns.histplot(data=train_df, x='Age', hue='Survived', bins = 40, kde=True)


# - Majority passengers were from 18-40 ages 
# - Children had more chance to survive than other ages

# #### Fare

# In[32]:


train_df['Fare'].describe()


# In[34]:


sns.histplot(data=train_df, x='Fare',hue = 'Survived', bins = 40, palette='Blues')


# In[38]:


#To name for 0-25% quartile, 25-50, 50-75, 75-100 (chia giá vé ra thành 4 khoảng, và đặt tên cho từng khoảng)
fare_categories = ['Economics', 'Standard', 'Expensive', 'Luxury']
quartile_data = pd.qcut(train_df['Fare'], 4, labels=fare_categories)

sns.countplot(x=quartile_data, hue=train_df['Survived'], palette='Blues')


# - Distribution of Fare
#     - Fare does not follow a normal distribution and has a huge spike at the price range `[0-$100]`
#     - The distribution is left-skewed with 75% of the fair paid under `$31` and a max paid fare of `$512`
# - Quartile plot:
#     - Passenger with Expensive and Luxury Fare Class had higher survival opportunity rather than Economics and Standard

# In[36]:


train_df['Fare']

