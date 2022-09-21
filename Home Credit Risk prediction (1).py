#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
home_credit = pd.read_csv('application_train.csv')
home_credit.head()


# In[2]:


#Are there any missing data?
print(home_credit.isnull().sum())


# #### Finding columns who have more than 40% missing data and dropping them from the dataset

# In[3]:


def find_missing(data):
    missing_count = data.isnull().sum().values
    total = len(data)
    ratio_missing = missing_count/total*100
    return pd.DataFrame(data={'column name':data.columns.values, 'missing_ratio':ratio_missing})


# In[4]:


#There aren't any columns with more than 40% missing data
find_missing(home_credit).sort_values(['missing_ratio'], ascending=False).head(30)


# In[ ]:


#If there are columns with more than 40% missing data, then run these codes
#missingcolumns_40=find_missing(home_credit)
#missingcolumn_40_list=list(missingcolumns_40['column name'][missingcolumns_40.missing_ratio>40])
#len(missingcolumn_40_list)
#home_credit.drop(missingcolumns_40_list,axis=1,inplace=True)


# In[5]:


#Draw bar chart to see default distribution
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
print(home_credit.TARGET.value_counts())
home_credit.TARGET.value_counts().plot(kind='bar')


# In[6]:


#Draw a bar chart to see contract type distribution 
print(home_credit.NAME_CONTRACT_TYPE.value_counts())
home_credit.NAME_CONTRACT_TYPE.value_counts().plot(kind='bar')


# In[7]:


pd.crosstab(home_credit['TARGET'], home_credit['NAME_CONTRACT_TYPE'])


# In[8]:


print(pd.crosstab(home_credit['CODE_GENDER'], home_credit['NAME_CONTRACT_TYPE'], margins=True))


# In[9]:


print(home_credit.CODE_GENDER.value_counts())
home_credit.CODE_GENDER.value_counts().plot(kind='bar')


# ###### Removing the 4 rows who have gender as XNA in them for the purpose of consistency

# In[10]:


home_credit = home_credit[home_credit.CODE_GENDER!='XNA']


# In[11]:


print(home_credit.CODE_GENDER.value_counts())
home_credit.CODE_GENDER.value_counts().plot(kind='bar')


# ###### There are 202448 female customers and 105059 male customers

# In[12]:


print(pd.crosstab(home_credit['CODE_GENDER'], home_credit['TARGET'], margins=True))


# ###### The cases of default recorded in female customers higher than that recorded in male customers. It is not a strage thing since the amount of female customers is twice as high as the amount of male customers 

# ### Use One-Hot Encoding

# In[13]:


#one-hot encoding of categorical variables using get dummies
home_credit = pd.get_dummies(home_credit)
home_credit


# ###### Now we find correlations of each variables with our TARGET variable to understand which variables affect the most in a positive or negative way

# In[14]:


#Find correlations with the target 
correlations = home_credit.corr()['TARGET'].sort_values()
correlations.head()


# In[15]:


#Display correlations
print('10 Most Positive Correlations:\n', correlations.tail(10))
print('\n10 Most Negative Correlations:\n', correlations.head(10))


# ###### The target value correlated with the days_birth in the first positive correlation. However, since the days_birth is a negative value, it means it is negatively correlated, which means that the smaller your age is, the more likely you're going to default

# In[16]:


import seaborn as sns


# In[17]:


#Calcute customer's age in days at the time of application
print(home_credit.loc[home_credit['TARGET'] == 0, 'DAYS_BIRTH'] / 365*-1)


# In[18]:


plt.figure(figsize = (8, 8))

#Distribution plot of loans that were repaid on time
sns.distplot(home_credit.loc[home_credit['TARGET'] == 0, 'DAYS_BIRTH']/365*-1, hist=False, label = 'target = 0')

#Distribution plot of loans that were not repaid on time
sns.distplot(home_credit.loc[home_credit['TARGET'] == 1, 'DAYS_BIRTH']/365*-1, hist=False, label = 'target = 1')

#Labeling of plot
plt.xlabel('Age (years)')
plt.ylabel('Density')
plt.title('Distribution of Ages')
plt.legend()


# In[19]:


n, bins, patches = plt.hist(x=home_credit['DAYS_BIRTH']/365*-1,
                            bins='auto', color='blue',alpha=0.7, rwidth=0.85)
plt.title("Age Distribution")
plt.show()


# ###### Plotting a similar plot for NAME_EDUCATION_TYPE_Higher education

# In[20]:


print(pd.crosstab(home_credit['NAME_EDUCATION_TYPE_Higher education'],
                  home_credit['TARGET'], margins=True))


# In[21]:


plt.figure(figsize = (8, 8))

#Distribution plot of loans that were repaid on time
sns.distplot(home_credit.loc[home_credit['TARGET'] == 0, 'NAME_EDUCATION_TYPE_Higher education'], hist=False, label = 'target = 0')

#Distribution plot of loans that were not repaid on time
sns.distplot(home_credit.loc[home_credit['TARGET'] == 1, 'NAME_EDUCATION_TYPE_Higher education'], hist=False, label = 'target = 1')

#Labeling of plot
plt.xlabel('Highest Education')
plt.ylabel('Density')
plt.title('Distribution of Education')
plt.legend()


# ###### After applying one-hot encoding, 1 stands for Higher Education, and 0 stands for other types of education (including Academic Degree, Incomplete higher, Lower Secondary, Secondary/secondary special). The above chart doesn't tell us much information about the relationship between customer's highest education and the likely to default

# In[22]:


plt.figure(figsize = (8, 8))

#Distribution plot of loans that were repaid on time
sns.distplot(home_credit.loc[home_credit['TARGET'] == 0, 'CODE_GENDER_M'], hist=False, label = 'target = 0')

#Distribution plot of loans that were not repaid on time
sns.distplot(home_credit.loc[home_credit['TARGET'] == 1, 'CODE_GENDER_M'], hist=False, label = 'target = 1')

#Labeling of plot
plt.xlabel('CODE_GENDER')
plt.ylabel('Density')
plt.title('Distribution of Gender')
plt.legend()


# ###### After using one-hot encoding, 1 stands for male and 0 stands for female. This chart tells us that: if you are female, you are more likely to repay the loan on time. However, I don't consider this is a believable result, since the amount of female customers is twice as high as the amount of male customers

# ### Population Level Predictions

# In[23]:


home_credit_onehot = home_credit


# In[24]:


#impute missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = 'median')
imputer.fit(home_credit)
data_complete = imputer.transform(home_credit)
print(data_complete.shape)


# In[29]:


data_complete = pd.DataFrame(data_complete, columns = home_credit_onehot.columns)
data_complete


# In[31]:


data_target_0 = data_complete[data_complete["TARGET"] == 0]
data_target_0


# In[32]:


data_target_1 = data_complete[data_complete["TARGET"]==1]
data_target_1


# ## Imbalaced data 
# There's a great difference in the amount of target 0 (282,686 cases) and target 1 (24,825 cases). A classification data set with skewed class proportions is called imbalanced. Classes that make up a large proportion of the data set are called majority classes (target 0). Those that make up a smaller proportion are minority classes (target 1). In order to run a correct machine learning model, the proportion of these two classes must be the same. 
# 
# In this project, I will use this method:

# ####  Just take 25,000 first rows of target 0, target 1 stays the same

# In[33]:


modified_data = pd.concat([data_target_0[:25000], data_target_1])
from sklearn.utils import shuffle
modified_data = shuffle(modified_data)
data_complete1 = modified_data
data_complete1


# In[34]:


# Split the dataset
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(data_complete1.drop(['TARGET'], axis=1), data_complete1['TARGET'], test_size=0.3, random_state=42)


# ### Logistic Regression Model 

# Now, we run a basic logistic regression by just using the normalized data as a baseline model

# In[35]:


from sklearn.linear_model import LogisticRegression
logit = LogisticRegression()
logit.fit(train_X, train_y)
#Predicting the model
pred_logit = logit.predict(test_X)


# In[36]:


from sklearn.metrics import classification_report, accuracy_score, confusion_matrix,
roc_auc_score,plot_confusion_matrix, plot_precision_recall_curve


# In[37]:


print('The accuracy of logit model is:', accuracy_score(test_y, pred_logit))
print(classification_report(test_y, pred_logit))


# - Precision: Out of all people that the model predicted would be defaulted, 88% actually did.
# - Recall: Out of all people that actually were defaulted, the model predicted this outcome correctly for 85% of those players.
# - Since F1-score is 0.86, close to 1, it tells us that the model does a good job of predicting whether or not people would be defaulted.
# - Among the people in the test dataset, 7547 people were not defaulted and 7401 people were defaulted.

# ### Random Forest Classifier

# In[38]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
#Fitting the model
rf.fit(train_X, train_y)
#Predicting the model
pred_rf = rf.predict(test_X)


# In[39]:


#Evaluating the Random Forest model

print('The accuracy of random forest model is:',
accuracy_score(test_y, pred_rf))
print(classification_report(test_y, pred_rf))


# ### Extreme Gradient Boosting (XGBoost)

# In[41]:


pip install xgboost


# In[42]:


import xgboost as xgb
xgb_clf = xgb.XGBClassifier()
#fitting the model
xgb_clf.fit(train_X, train_y)
#predicting the model
xgb_predict = xgb_clf.predict(test_X)


# In[43]:


#Evaluating the xgboost model
print("The accuracy of xgboost model is:",
     accuracy_score(test_y, xgb_predict))
print(classification_report(test_y, xgb_predict))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




