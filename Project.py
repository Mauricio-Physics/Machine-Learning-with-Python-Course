#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


movies = pd.read_csv('C:/Users/MAURICIO/Desktop/Peliculas.csv',encoding='utf-8')


# In[3]:


movies.head()                                               ## Five first elements


# In[4]:


movies.shape                                            ## Size of the data


# In[5]:


movies.columns                                          ## Show Columns


# In[6]:


movies.index                                            ## Show Index


# In[7]:


c1 = movies['movie_title']                              ## Work with just one column


# In[8]:


r1 = movies.loc[10,:]                                   ## Work with just one row              


# In[9]:


movies.info()                                           ## Amount of elements different to zero


# In[10]:


movies.dtypes                                                    ## Type of elements of each column


# In[11]:


(movies.dtypes == 'float64') | (movies.dtypes == 'int64')                          ## Number type


# In[12]:


movies.dtypes == 'object'                                                        ## Object type


# In[13]:


num_cols = [i for i in range(len(movies.dtypes)) if movies.dtypes[i] != 'object']          
num_cols                                            ## Substracting the indexes of movies types which are numbers


# In[14]:


num_col = [movies.columns[j] for j in num_cols]
num_col                                             ## Taking the categories


# In[15]:


obj_cols = [i for i in range(len(movies.dtypes)) if movies.dtypes[i] == object]
obj_cols                                            ## Substracting the indexes of movies types which are object


# In[16]:


obj_col = [movies.columns[j] for j in obj_cols]
obj_col                                             ## Taking the categories


# In[17]:


movies_num = movies[num_col]


# In[18]:


movies_num.describe()                                               ## Statistics about these columns


# In[19]:


movies_num['title_year'].hist()


# In[20]:


movies_num['imdb_score'].hist()


# In[21]:


movies_num['budget'].hist()


# In[22]:


mask = (movies_num['budget'] > 1e9)


# In[23]:


movies[mask]


# In[24]:


financials = pd.read_csv('C:/Users/MAURICIO/Desktop/thenumbers.csv',encoding='utf-8')               ## New data set
financials.shape


# In[25]:


f = financials[['movie_title','production_budget','worldwide_gross']]


# In[26]:


f.shape


# In[27]:


m_n = pd.concat([movies['movie_title'],movies_num],axis=1)                              ## Concatenation


# In[28]:


movies_v2 = pd.merge(f,m_n,on='movie_title',how='left')                                 ## Merge
movies_v2.shape


# In[29]:


movies_v2


# In[ ]:





# In[30]:


## Missing Data


# In[31]:


movies_v2.notnull().apply(pd.Series.value_counts)               ## not null: missing data
                                                                ## value counts: count the values with some specific condition


# In[32]:


(movies_v2 != 0).apply(pd.Series.value_counts)                 ## Data different to zero


# In[33]:


available = ((movies_v2 != 0) & (movies_v2.notnull()))                   ## Both Conditions


# In[34]:


available.all(axis=1).value_counts()


# In[35]:


mask = available['worldwide_gross']                                        
movies_v2 = movies_v2[mask]                                                   ## Movies_v2 es solo los datos que estan.
movies_v2.shape                                                               ## 4104 data available about worldwide_gross


# In[36]:


((movies_v2 != 0) & (movies_v2.notnull())).worldwide_gross.value_counts()         ## Not missing or null data


# In[37]:


imputer = SimpleImputer(missing_values=np.nan, strategy='mean')                   ## Fill the data missing with the mean


# In[38]:


movies_v2 = movies_v2.drop('movie_title',axis=1)
movies_v2 = movies_v2.drop('duration',axis=1)


# In[39]:


movies_v2.head()


# In[40]:


movies_v2.values                                                       ## Values movies


# In[41]:


values = imputer.fit_transform(movies_v2)                             ## Values movies fixed


# In[42]:


values


# In[43]:


X = pd.DataFrame(values)                                     ## Creating the dataframe
X.columns = movies_v2.columns                                ## Creating columns 
X.index = movies_v2.index                                    ## Creating rows
X.head()
print(len(X))


# In[44]:


X.to_csv('C:/Users/MAURICIO/Desktop/First_Result.csv',index=False)                ## Saving the data


# In[45]:


x = pd.read_csv('C:/Users/MAURICIO/Desktop/First_Result.csv',encoding='utf-8')    ## Reading data
x = x.drop(['gross'], axis=1)      


# In[46]:


y = x['worldwide_gross']                                                      ## money to gain


# In[47]:


x = x.drop(['worldwide_gross'], axis=1)                                         ## It is because that is what I want to predict


# In[48]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.4,random_state=1)     ## Spliting the data in test and train (60% Training-40% Test)
print(len(x))
print(len(x_train))
print(len(x_test))                                                           


# In[49]:


model = Lasso()
model.fit(x_train,y_train)                                                   ## Training part


# In[50]:


predicted = model.predict(x_test)                                            ## Making the prediction 


# In[51]:


plt.hist([predicted,y_test]);                                                ## Comparation


# In[52]:


model.score(x_test,y_test)


# In[53]:


residuals = y_test - predicted                                                ## Errors 


# In[54]:


plt.scatter(y_test,np.abs(residuals))


# In[55]:


ab_residuals = np.abs(residuals) / y_test                                    ## Percentage
plt.scatter(y_test,ab_residuals)


# In[56]:


lab_residuals = np.log(ab_residuals)
plt.scatter(y_test,lab_residuals)


# In[57]:


x.corr()                                                                   ## Correlation between variables


# In[58]:


get_ipython().run_line_magic('matplotlib', 'inline')
sb.heatmap(x.corr())


# In[59]:


model.coef_


# In[60]:


len(model.coef_)
print(x.columns)


# In[61]:


var = np.floor(np.log10(np.abs(model.coef_)))                                           ## Entire part
plt.figure(figsize=(15,6))
plt.plot(x.columns,var)


# In[ ]:





# In[62]:


## Let´s do it by default


# In[63]:


xx_train, xx_test, yy_train, yy_test = train_test_split(x,y)


# In[64]:


len(xx_train)/len(x)


# In[65]:


model.fit(xx_train,yy_train)


# In[66]:


model.score(xx_test,yy_test)


# In[67]:


model.coef_


# In[68]:


var_ = np.floor(np.log10(np.abs(model.coef_)))
plt.figure(figsize=(15,6))
plt.plot(x.columns,var_)


# In[ ]:





# In[69]:


## Correlation between variables


# z = pd.concat([x,y],axis=1)
# sb.pairplot(z)

# In[70]:


z = pd.concat([x,y],axis=1)
sb.heatmap(z.corr())


# In[71]:


selector = SelectKBest(mutual_info_regression, k=4)                               ## Select the best feautures
selector.fit(x,y)


# In[72]:


scores = selector.scores_
plt.figure(figsize=(15,6))
plt.plot(x.columns,scores)


# In[ ]:





# In[73]:


## Create 3 different models reducing the features to visualize the changings


# In[74]:


x_2 = x[['production_budget','title_year','duration.1','cast_total_facebook_likes','imdb_score']]
x_3 = x[['production_budget','cast_total_facebook_likes','imdb_score']]
x_4 = x[['production_budget','cast_total_facebook_likes','budget']]


# In[75]:


c2 = ['production_budget','title_year','duration.1','cast_total_facebook_likes','imdb_score']
x2_train, x2_test, y2_train, y2_test = x_train[c2], x_test[c2], y_train, y_test

c3 = ['production_budget','cast_total_facebook_likes','imdb_score']
x3_train, x3_test, y3_train, y3_test = x_train[c3], x_test[c3], y_train, y_test

c4 = ['production_budget','cast_total_facebook_likes','budget']
x4_train, x4_test, y4_train, y4_test = x_train[c4], x_test[c4], y_train, y_test


# In[76]:


model2 = Lasso()
model3 = Lasso()
model4 = Lasso()


# In[77]:


model2.fit(x2_train,y2_train)
model3.fit(x3_train,y3_train)
model4.fit(x4_train,y4_train)


# In[78]:


print(model2.score(x2_test,y2_test))
print(model3.score(x3_test,y3_test))
print(model4.score(x4_test,y4_test))


# In[ ]:





# In[79]:


## Rescale the data


# In[80]:


scaler = StandardScaler()
scaler.fit(x_train)
print(scaler.mean_)                             ##   Mean
print(scaler.scale_)                            ##   Standard Desviation


# In[81]:


x_train_scaled, x_test_scaled = (scaler.transform(x_train), scaler.transform(x_test))      ## Rescale the data


# In[82]:


model_scaled = Lasso()
model_scaled.fit(x_train_scaled,y_train)


# In[83]:


print(model.score(x_test,y_test))
print(model_scaled.score(x_test_scaled,y_test))


# In[85]:


model_scaled = make_pipeline(StandardScaler(),Lasso())                                ## Faster method
model_scaled.fit(x_train,y_train)
print(model_scaled.score(x_test,y_test))


# In[ ]:




