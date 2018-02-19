
# coding: utf-8

# In[136]:


import pandas as pd
import numpy as np
from itertools import cycle, islice
import math


# In[98]:


data = pd.read_csv("/Users/deepikamulchandani/Downloads/DataforLR.csv")


# In[99]:


data


# In[100]:


import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score


# In[101]:


data['y']=0


# In[102]:


def calc_resp():
    for row in data.itertuples():
        if(row[30]=='left'):
            data.set_value(row.Index,'y',1)

calc_resp()


# In[103]:


data


# In[104]:


X = pd.DataFrame.copy(data)


# In[105]:


X


# In[106]:


del X['Y']


# In[107]:


X


# In[108]:


del X['y']


# In[109]:


X


# In[110]:


X[['NumericSupervisor']]=X[['supervisor']].stack().rank(method='dense').unstack()


# In[111]:


X


# In[112]:


y = data['y']


# In[113]:


y


# In[114]:


del X['supervisor']


# In[115]:


model = LogisticRegression()
model = model.fit(X, y)


# In[116]:


model.score(X, y)


# In[117]:


y.mean()


# In[118]:


var_imp=pd.DataFrame(zip(X.columns, np.transpose(model.coef_)),columns=['Variables', 'Importance'])


# In[119]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
model2 = LogisticRegression()
model2.fit(X_train, y_train)


# In[120]:


predicted = model2.predict(X_test)
print predicted


# In[121]:


probs = model2.predict_proba(X_test)
print probs


# In[122]:


print metrics.accuracy_score(y_test, predicted)
print metrics.roc_auc_score(y_test, probs[:, 1])


# In[123]:


print metrics.confusion_matrix(y_test, predicted)
print metrics.classification_report(y_test, predicted)


# In[124]:


scores = cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10)
print scores
print scores.mean()


# In[125]:


import matplotlib.pyplot as plt


# In[126]:


var_imp['Importance']=var_imp['Importance'].astype(float)


# In[31]:


#var_imp['Importance']= var_imp['Importance'].abs()


# In[128]:


var_imp


# In[129]:


var_imp['Importance']


# In[137]:


var_imp['Normalized_LR']=0.0
def sum_squared_normalization():
    val=0
    for row in var_imp.itertuples():
        val = val + row[2]*row[2]
    val2 = math.sqrt(val)
    for row in var_imp.itertuples():
        var_imp.set_value(row.Index, 'Normalized_LR', row[2]/val2)
        
sum_squared_normalization()


# In[139]:


var_imp['Normalized_LR']= var_imp['Normalized_LR'].abs()
var_imp


# In[33]:


my_colors = list(islice(cycle(['b', 'r', 'g', 'y', 'k']), None, len(var_imp)))
ax=var_imp.plot(kind='bar', title ="Variable Importance", figsize=(15, 10), legend=True, fontsize=12,color=my_colors)
ax.set_xlabel("Variables", fontsize=12)
ax.set_xticklabels(var_imp['Variables'])
ax.set_ylabel("Importance", fontsize=12)


# In[140]:


var_imp3 = pd.read_csv("/Users/deepikamulchandani/Downloads/VariableImportance.csv")


# In[141]:


var_imp3


# In[142]:


del var_imp3['Unnamed: 0']


# In[145]:


var_imp4 = pd.DataFrame.copy(var_imp3)


# In[146]:


var_imp4


# In[147]:


#plt.show()


# In[148]:


var_imp4['Importance'] = var_imp['Normalized_LR']


# In[149]:


var_imp4


# In[150]:


var_imp['Importance_SVM']=var_imp3['Importance_SVM']


# In[151]:


var_imp


# In[152]:


var_imp['Normalized_SVM']=0.0
def sum_squared_normalization():
    val=0
    for row in var_imp.itertuples():
        val = val + row[4]*row[4]
    val2 = math.sqrt(val)
    for row in var_imp.itertuples():
        var_imp.set_value(row.Index, 'Normalized_SVM', row[4]/val2)
        
sum_squared_normalization()


# In[153]:


var_imp


# In[35]:


var_imp


# In[154]:


var_imp4['Importance_SVM']=var_imp['Normalized_SVM']


# In[155]:


var_imp4


# In[36]:


var_imp['Importance_RF']=0.0


# In[85]:


var_imp.iloc[24,2]=0.0731


# In[38]:


var_imp


# In[39]:


var_imp['Importance_SVM']=0.0


# In[67]:


var_imp.iloc[27,3]=0.00078


# In[219]:


var_imp3


# In[87]:


var_imp.to_csv("/Users/deepikamulchandani/Downloads/VariableImportance.csv")


# In[174]:


var_imp2 = var_imp4[(var_imp4['Importance']>0.01)&(var_imp4['Importance_RF']>0.01)&(var_imp4['Importance_SVM']>0.01)]


# In[175]:


var_imp2


# In[209]:


var_imp2.iloc[6,0]='Email sentiment-negative outliers'


# In[210]:


var_imp2


# In[245]:


var_imp2.iloc[6,0]='Email sentiment -ve outliers'


# In[270]:


import numpy as np
import matplotlib.pyplot as plt
 
# data to plot
n_groups = 7
imp1 = var_imp2['Importance']
imp2 = var_imp2['Importance_RF']
imp3 = var_imp2['Importance_SVM']
 
# create plot
fig, ax = plt.subplots(figsize=(30,10))
index = np.arange(n_groups)
bar_width = 0.25
opacity = 0.8
 
rects1 = plt.bar(index, imp1, bar_width,
                 alpha=opacity,
                 color='c',
                 label='Logistic Regression',
                )
 
rects2 = plt.bar(index + bar_width, imp2, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Random Forest',
                )
 
rects3 = plt.bar(index + bar_width + bar_width, imp3, bar_width,
                 alpha=opacity,
                 color='y',
                 label='SVM',
                )

plt.xlabel('PREDICTORS',fontsize=20)
plt.ylabel('IMPORTANCE',fontsize=20)
plt.xticks(index + bar_width, var_imp2['Variables'], fontsize=17)
plt.yticks(fontsize=17)
plt.legend(fontsize=20)

plt.show()


# In[271]:


n_groups = 29
imp1 = var_imp3['Importance_RF']

fig, ax = plt.subplots(figsize=(30,10))
index = np.arange(n_groups)
bar_width = 0.65
opacity = 0.8
 
rects1 = plt.bar(index, imp1, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Predictors')

plt.xlabel('PREDICTORS',fontsize=20)
plt.ylabel('IMPORTANCE',fontsize=20)
plt.xticks(index, var_imp3['Variables'], fontsize=17, rotation=70)
plt.yticks(fontsize=20)
plt.legend(fontsize=20)

plt.show()

