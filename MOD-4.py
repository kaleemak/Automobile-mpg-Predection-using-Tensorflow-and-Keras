#!/usr/bin/env python
# coding: utf-8

# In[1]:


#this module is about early stopping and the feature encoding vectors,or preprocessing


# In[36]:


#dealing with outliers
import pandas as pd
import numpy as np
from sklearn import preprocessing 
from sklearn import metrics
from scipy.stats import zscore


# In[37]:


#the dataset preperation
df =pd.read_csv('auto-mpg.csv',na_values =['NA,','?'])


# In[38]:


#fill the missing values
med =df['horsepower'].median()
df['horsepower'] = df['horsepower'].fillna(med)


# In[39]:


#drop the name column
df.drop('name',axis=1,inplace=True)


# In[40]:


df.head()


# In[41]:


#remove the outliers from horsepower
def remove_outliers(df, name, sd):
    drop_rows = df.index[(np.abs(df[name] - df[name].mean())
                          >= (sd * df[name].std()))]
    df.drop(drop_rows, axis=0, inplace=True)
print('before ',len(df))
remove_outliers(df,'mpg',2)
print('after',len(df))


# In[67]:


#Early Stopping on iris dataset
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from keras.layers.core import Dense,Activation
from keras.callbacks import EarlyStopping
from sklearn import preprocessing
from keras.models import Sequential
from sklearn import metrics
from sklearn.model_selection import train_test_split
#get the data
df =pd.read_csv('iris.csv',na_values=['NA','?'])
def encode_text_index(df, name):
    le = preprocessing.LabelEncoder()
    df[name] = le.fit_transform(df[name])
    return le.classes_
species= encode_text_index(df,'species')
def to_xy(df, target):
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)
    # find out the type of the target column.  Is it really this hard? :(
    target_type = df[target].dtypes
    target_type = target_type[0] if hasattr(
        target_type, '__iter__') else target_type
    # Encode to int for classification, float otherwise. TensorFlow likes 32 bits.
    if target_type in (np.int64, np.int32):
        # Classification
        dummies = pd.get_dummies(df[target])
        return df[result].values.astype(np.float32), dummies.values.astype(np.float32)
    # Regression
    return df[result].values.astype(np.float32), df[[target]].values.astype(np.float32)
x,y =to_xy(df,'species')
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.25,random_state=42)
#define the model

model =Sequential()
#add the layers on that model
model.add(Dense(25,input_dim =x.shape[1],activation ='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(y.shape[1],activation='softmax'))
#compile that model
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
#ddefien the early stopping
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')

model.fit(x_train, y_train,validation_data=(x_test,y_test),epochs=1000,callbacks=[monitor],verbose=2)


# In[69]:


pred=model.predict(x_test)
print(pred[0:5])


# In[70]:


pred =np.argmax(pred,axis=1)


# In[72]:


pred[0:5]


# In[74]:


ycompare = np.argmax(y_test,axis=1)
ycompare[0:5]


# In[90]:


#regression on auto-mpg.csv
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from keras.layers.core import Dense,Activation
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn import metrics
from scipy.stats import zscore
df =pd.read_csv('auto-mpg.csv',na_values=['NA','?'])
med =df['horsepower'].median()
df['horsepower'] = df['horsepower'].fillna(med)
df.drop('name',axis=1,inplace=True)
x,y =to_xy(df,'mpg')
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.25,random_state=42)
model =Sequential()
model.add(Dense(25,input_dim=x.shape[1],activation='relu'))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='Adam',metrics=['accuracy'])
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
model.fit(x_train,y_train,validation_data=(x_test,y_test),callbacks=[monitor],verbose=2,epochs=1000)


# In[97]:


pred =model.predict(x_test)


# In[106]:


pred[0:5]


# In[107]:


y_test[0:5]


# In[135]:


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from keras.layers.core import Dense,Activation
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
from sklearn import metrics
from scipy.stats import zscore
df =pd.read_csv('auto-mpg.csv',na_values=['NA','?'])
#shuffling
np.random.seed(42)
#now the shuffiling
df =df.reindex(np.random.permutation(df.index))
#to order the index for the convenient,if we cannot use it the row indexes are changed
df.reset_index(inplace =True ,drop =True)
df.drop('name',axis=1,inplace=True)
med =df['horsepower'].median()
df['horsepower'] =df['horsepower'].fillna(med)
x,y =to_xy(df,'mpg')
kf =KFold(5)
real_y = []
prediction =[]
fold=0
for train,test in kf.split(x):
    fold+=1
    print("Fold #{}".format(fold))
    x_train = x[train]
    y_train =y[train]
    x_test = x[test]
    y_test = y[test]
    #define the model
    model =Sequential()
    #add the layers
    model.add(Dense(25,input_dim=x.shape[1],activation='relu'))
    model.add(Dense(10,activation='relu'))
    model.add(Dense(y.shape[1]))
    #compile the model
    model.compile(loss='mean_squared_error',optimizer='Adam',metrics=['accuracy'])
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
    #fit the model
    model.fit(x_train,y_train,validation_data=(x_test,y_test),callbacks=[monitor],verbose=0,epochs=1000)
    pred =model.predict(x_test)
    real_y.append(y_test)
    prediction.append(pred)
    score = np.sqrt(metrics.mean_squared_error(pred,y_test))
    print("Fold score (RMSE): {}".format(score))
real_y = np.concatenate(real_y)
prediction = np.concatenate(prediction)
score = np.sqrt(metrics.mean_squared_error(prediction,real_y))
print("Final, out of sample score (RMSE): {}".format(score))

#write the validation sample for the kagle competation
real_y = pd.DataFrame(real_y)
prediction =pd.DataFrame(prediction)
newDF = pd.concat([df,real_y,prediction])
newDF.to_csv('mynewdataframe.csv',index =False)


# In[137]:


#handouts for the validation,before the real_world use
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from keras.layers.core import Dense,Activation
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
from sklearn import metrics
from scipy.stats import zscore

from sklearn.model_selection import train_test_split
df =pd.read_csv('auto-mpg.csv',na_values=['NA','?'])
#shuffling
np.random.seed(42)
#now the shuffiling
df =df.reindex(np.random.permutation(df.index))
#to order the index for the convenient,if we cannot use it the row indexes are changed
df.reset_index(inplace =True ,drop =True)
df.drop('name',axis=1,inplace=True)
med =df['horsepower'].median()
df['horsepower'] =df['horsepower'].fillna(med)
x,y =to_xy(df,'mpg')
x_main,x_handouts,y_main,y_handouts =train_test_split(x,y,test_size=0.10)#10% used for the handouts

kf =KFold(5)
real_y = []
prediction =[]
fold=0
for train,test in kf.split(x_main):
    fold+=1
    print("Fold #{}".format(fold))
    x_train = x_main[train]
    y_train =y_main[train]
    x_test = x_main[test]
    y_test = y_main[test]
    #define the model
    model =Sequential()
    #add the layers
    model.add(Dense(25,input_dim=x.shape[1],activation='relu'))
    model.add(Dense(10,activation='relu'))
    model.add(Dense(y.shape[1]))
    #compile the model
    model.compile(loss='mean_squared_error',optimizer='Adam',metrics=['accuracy'])
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
    #fit the model
    model.fit(x_train,y_train,validation_data=(x_test,y_test),callbacks=[monitor],verbose=0,epochs=1000)
    pred =model.predict(x_test)
    real_y.append(y_test)
    prediction.append(pred)
    score = np.sqrt(metrics.mean_squared_error(pred,y_test))
    print("Fold score (RMSE): {}".format(score))
real_y = np.concatenate(real_y)
prediction = np.concatenate(prediction)
score = np.sqrt(metrics.mean_squared_error(prediction,real_y))
print("Final, out of sample score (RMSE): {}".format(score))
#handout_prediction
handout_prediction = model.predict(x_handouts)
score = np.sqrt(metrics.mean_squared_error(handout_prediction,y_handouts))
print("Holdout score (RMSE): {}".format(score))

#write the validation sample for the kagle competation
real_y = pd.DataFrame(real_y)
prediction =pd.DataFrame(prediction)
newDF = pd.concat([df,real_y,prediction])
newDF.to_csv('mynewdataframe.csv',index =False)


# In[ ]:


#error calculation from scratch

