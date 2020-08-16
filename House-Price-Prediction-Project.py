#!/usr/bin/env python
# coding: utf-8

# In[355]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')
plt.style.use('ggplot')


# In[356]:


# Read data from current directory
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')


# In[357]:


# show data
train


# In[358]:


train_df.info()


# In[359]:


def get_object_cols(df):
    return list(df.select_dtypes(include='object').columns)

def get_numerical_cols(df):
    return list(df.select_dtypes(exclude='object').columns)


# In[360]:


list1 = ['Alley','Utilities', 'PoolQC', 'Fence', 'MiscFeature']
for item in list1:
    train.drop(columns=item, inplace=True)
    test.drop(columns=item, inplace=True)


# In[361]:


dict={'Y' : 1, 'N' : 0, 'Ex': 1, 'Gd' : 2, 'TA' :3, 'Fa' : 4, 'Po' : 5,  
     'GLQ' : 1, 'ALQ' : 2, 'BLQ' : 3, 'Rec' : 4, 'LwQ' : 5, 'Unf' : 6, 'NA' :7,
     'Gd' : 1 , 'Av' :2, 'Mn' : 3, 'No' :4, 'Gtl' : 1, 'Mod' : 2, 'Sev' :3,
      'Reg' : 1, 'IR1' :2, 'IR2' :3, 'IR3' :4}

## selecting columns for mapping dict
# 'RL':1, 'RM':2,'FV':3,'RH':4,'C (all)':5, 'Pave':1, 'Grvl':2,'Lvl':1,'Bnk':1,'HLS':2,'Low':3,'Inside':1, 'Corner':2,'CulDSac':3,
# 'FR2':4, 'FR3':5, 'Y':1, 'N':2, 'P':3,'Norm':1, 'Feedr':2, 'PosN':3, 'Artery':4, 'RRAe':5, 'RRNn':6, 'RRAn':7, 'PosA':8,'RRNe':9 
cols=['KitchenQual','LotShape','LandSlope','HeatingQC','FireplaceQu','ExterQual','ExterCond','BsmtQual',
     'BsmtFinType2','BsmtFinType1','BsmtExposure','BsmtCond','CentralAir']
for i in cols:
    train[i]=train[i].map(dict)


# In[362]:



##train["MSZoning"]= train["MSZoning"].replace("RL", 1)
#train["MSZoning"]= train["MSZoning"].replace("RM", 2)
##train["MSZoning"]= train["MSZoning"].replace("FV", 3)
#train["MSZoning"]= train["MSZoning"].replace("RH", 4)
#train["MSZoning"]= train["MSZoning"].replace("C (all)", 5)##
train.GarageCond.value_counts(dropna=False)


# In[363]:


dict = {'RL' :1, 'RM' :2,'FV':3,'RH' :4,'C (all)' :5, 'Pave' :1, 'Grvl' :2,
        'Lvl' :1,'Bnk' :1,'HLS':2,'Low' :3,'Inside' :1, 'Corner' :2,'CulDSac' :3,
       'FR2' :4, 'FR3' :5, 'Y':1, 'N' :2, 'P':3,'Norm' :1, 'Feedr' :2, 'PosN' :3, 'Artery' :4,
        'RRAe' :5, 'RRNn' :6, 'RRAn' :7, 'PosA':8,'RRNe' :9, 'TA': 1, 'Fa':2, 'Gd':3, 'Po':4, 'Ex':5}
cols=['Street','GarageQual','MSZoning', 'LandContour', 'Condition1', 'Condition2', 'GarageCond']
# 
for i in cols:
    train[i]=train[i].map(dict)


# In[364]:


train


# In[365]:


train.info()


# In[366]:


object_cols_train = get_object_cols(train)
# train numerical cols
numerical_cols_train = get_numerical_cols(train)


# In[367]:


object_cols_train


# In[240]:





# In[368]:


X_train=train.drop(['SalePrice'],axis=1)
y_train=train['SalePrice']
for item in object_cols_train:
    X_train.drop(columns=item, inplace=True)


# In[369]:


X_train


# In[370]:


# Create a model
xtrain, xtest, ytrain, ytest = train_test_split(X_train, y_train, test_size=0.20, random_state=42)
import xgboost as xgb
xgbr = xgb.XGBRegressor(verbosity=0) 
print(xgbr)


# In[371]:


xgbr.fit(xtrain, ytrain)


# In[372]:


score = xgbr.score(xtrain, ytrain)


# In[373]:


print("Training score: ", score)


# In[374]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(xgbr, xtrain, ytrain,cv=10)
print("Mean cross-validation score: %.2f" % scores.mean())


# In[375]:


y_pred=xgbr.predict(xtest)
y_pred


# In[376]:


ytest


# In[377]:


ytest-y_pred


# In[378]:


model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1,verbosity=0)
print(xgbr)


# In[379]:


model_xgb.fit(xtrain, ytrain)


# In[380]:


score = model_xgb.score(xtrain, ytrain)
print("Training score: ", score)


# In[381]:


scores = cross_val_score(model_xgb, xtrain, ytrain,cv=10)
print("Mean cross-validation score: %.2f" % scores.mean())


# In[382]:


y_pred=xgbr.predict(xtest)
y_pred


# In[383]:


y_pred=model_xgb.predict(xtest)
y_pred


# In[384]:


ytest


# In[385]:


ytest-y_pred


# In[392]:


xtrain.shape,ytrain.shape


# In[386]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(ytest, ypred)
print("MSE: %.2f" % mse)
print("RMSE: %.2f" % (mse*(1/2.0)))


# In[387]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(ytest, y_pred)
print("MSE: %.2f" % mse)
print("RMSE: %.2f" % (mse*(1/2.0)))


# In[402]:


vxtrain= xtrain.to_numpy()
vytrain= ytrain.to_numpy()
vxtest= xtest.to_numpy()
vytest= ytest.to_numpy()
vxtrain.shape


# In[403]:


vxtrain = vxtrain.reshape((vxtrain.shape[0], vxtrain.shape[1], 1))
print(vxtrain.shape)


# In[389]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


# In[404]:


# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(57, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(vxtrain, vytrain, epochs=100, batch_size=1, verbose=2)


# In[171]:


object_cols_train = get_object_cols(train)
# train numerical cols
numerical_cols_train = get_numerical_cols(train)


# In[179]:


for i in train.columns:
    print(train[i].value_counts(dropna=False))


# In[189]:


# 'RL':1, 'RM':2,'FV':3,'RH':4,'C (all)':5, 'Pave':1, 'Grvl':2,'Lvl':1,'Bnk':1,'HLS':2,'Low':3,'Inside':1, 'Corner':2,'CulDSac':3,
# 'FR2':4, 'FR3':5, 'Y':1, 'N':2, 'P':3,'Norm':1, 'Feedr':2, 'PosN':3, 'Artery':4, 'RRAe':5, 'RRNn':6, 'RRAn':7, 'PosA':8,'RRNe':9 

# MSSubClass LotArea LotShape LandContour Condition1 Condition2 GarageCond
# Check through each column using value_counts
# List of drop
#Alley Utilities PoolQC Fence MiscFeature

train.Street.value_counts(dropna=False)


# In[181]:


import collections
import itertools

def build_catalog(L):
    counter = itertools.count().next
    names = collections.defaultdict(counter)
    result = []
    for t in L:
        new_t = [ names[item] for item in t ]
        result.append(new_t)
    catalog = dict((name, idx) for idx, name in names.iteritems())
    return result, catalog


# In[182]:


outputlist, outputmapping = build_catalog(train['MSZoning'])


# In[132]:


# drop 5 conditions
list1 = ['Alley','Utilities', 'PoolQC', 'Fence', 'MiscFeature']
for item in list1:
    train.drop(columns=item, inplace=True)
    test.drop(columns=item, inplace=True)


# In[325]:


vtrain = train.to_numpy()
vtrain


# In[133]:


# Start preprocessing data set condition to number and set NaN to 0

train.MSZoning.value_counts(dropna=False)


# In[134]:


train["MSZoning"]= train["MSZoning"].replace("RL", 1)
train["MSZoning"]= train["MSZoning"].replace("RM", 2)
train["MSZoning"]= train["MSZoning"].replace("FV", 3)
train["MSZoning"]= train["MSZoning"].replace("RH", 4)
train["MSZoning"]= train["MSZoning"].replace("C (all)", 5)


# In[135]:


train.LotFrontage.value_counts(dropna=False)


# In[136]:


train["LotFrontage"]= train["LotFrontage"].fillna(0)


# In[137]:


train.LotShape.value_counts(dropna=False)


# In[138]:


train["LotShape"]= train["LotShape"].replace("Reg", 1)
train["LotShape"]= train["LotShape"].replace("IR1", 2)
train["LotShape"]= train["LotShape"].replace("IR2", 3)
train["LotShape"]= train["LotShape"].replace("IR3", 4)


# In[141]:


train.LandContour.value_counts(dropna=False)


# In[140]:


train["LandContour"]= train["LandContour"].replace("Lvl", 1)
train["LandContour"]= train["LandContour"].replace("Bnk", 2)
train["LandContour"]= train["LandContour"].replace("HLS", 3)
train["LandContour"]= train["LandContour"].replace("Low", 4)


# In[146]:


train.LotConfig.value_counts(dropna=False)


# In[147]:


train["LotConfig"]= train["LotConfig"].replace("Inside", 1)
train["LotConfig"]= train["LotConfig"].replace("Corner", 2)
train["LotConfig"]= train["LotConfig"].replace("CulDSac", 3)
train["LotConfig"]= train["LotConfig"].replace("FR2", 4)
train["LotConfig"]= train["LotConfig"].replace("FR3", 5)


# In[155]:



train.LandSlope.value_counts(dropna=False)


# In[154]:


replaceItem = {'Gtl': 1,'Mod': 2,'Sev':3} 
  
# traversing through dataframe  
# values where key matches 
train.LandSlope = [replaceItem[item] for item in train.LandSlope] 


# In[160]:


train.Neighborhood.value_counts(dropna=False)


# In[158]:


train.Neighborhood.unique()


# In[159]:


replaceItem = {'CollgCr':1, 'Veenker':2, 'Crawfor':3, 'NoRidge':4, 'Mitchel':5, 'Somerst':6,
       'NWAmes':7, 'OldTown':8, 'BrkSide':9, 'Sawyer':10, 'NridgHt':11, 'NAmes':12,
       'SawyerW':13, 'IDOTRR':14, 'MeadowV':15, 'Edwards':16, 'Timber':17, 'Gilbert':18,
       'StoneBr':19, 'ClearCr':20, 'NPkVill':21, 'Blmngtn':22, 'BrDale':23, 'SWISU':24,
       'Blueste':25} 
  
# traversing through dataframe  
# values where key matches 
train.Neighborhood = [replaceItem[item] for item in train.Neighborhood] 


# In[164]:



train.Condition1.value_counts(dropna=False)


# In[162]:


train.Condition1.unique()


# In[163]:


replaceItem = {'Norm':1, 'Feedr':2, 'PosN':3, 'Artery':4, 'RRAe':5, 'RRNn':6, 'RRAn':7, 'PosA':8,
       'RRNe':9}
# traversing through dataframe  
# values where key matches 
train.Condition1 = [replaceItem[item] for item in train.Condition1]


# In[48]:


train_df, validate_df = train_test_split(train, test_size=0.20, random_state=42)
train_df.shape


# In[114]:


train_df.Street.value_counts(dropna=False)


# In[5]:


# Look at the missing values
sns.heatmap(train.isnull(),yticklabels=False, cmap='plasma')


# In[30]:


# display column Alley including NaN
k= train.MiscFeature.value_counts(dropna=False)
train.columns


# In[16]:


train.GarageYrBlt.value_counts(dropna=False)


# In[37]:


# Set NaN to 0
list1 = ['BsmtQual', 'FireplaceQu', 'GarageType', 'GarageCond', 'GarageFinish', 'GarageQual', 'MasVnrType', 'MasVnrArea',
         'BsmtExposure','BsmtFinType2']

for item in list1:
    train[item] = train[item].fillna(train[item].mode()[0])
    test[item] = test[item].fillna(test[item].mode()[0])


# In[11]:


list1 = ['GarageYrBlt', 'PoolQC', 'Fence', 'MiscFeature']

for item in list1:
    train.drop(columns=item, inplace=True)
    test.drop(columns=item, inplace=True)


# In[16]:


train['BsmtQual']


# In[13]:


train


# In[ ]:




