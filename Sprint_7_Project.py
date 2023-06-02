#!/usr/bin/env python
# coding: utf-8

# # Sprint 7 Project

# ## Initializing Data

# In[1]:


import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict, RandomizedSearchCV, ParameterGrid
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error


# In[2]:


random_state = 54321
np.random.seed(random_state)
random.seed(random_state)


# In[3]:


df_user_behavior = pd.read_csv('https://code.s3.yandex.net/datasets/users_behavior.csv')
display(df_user_behavior.head(10))


# ## Exploratory Data Analysis

# In[4]:


display(df_user_behavior.describe())


# In[5]:


display(df_user_behavior.info())


# In[6]:


df_user_behavior.groupby('is_ultra').count().plot(
                     kind='pie',
                     y='mb_used',
                     autopct='%1.0f%%',
                     legend=False)
plt.title('Ultra or Not')
plt.ylabel('')
plt.show()


# In[7]:


df_user_behavior.groupby('is_ultra').count().plot(
                     kind='bar',
                     y='mb_used',
                     legend=False)
plt.title('Ultra or Not')
plt.ylabel('')
plt.xlabel('')
plt.show()


# In[8]:


df_user_behavior["is_ultra"].value_counts()/len(df_user_behavior)*100


# In[9]:


target = "is_ultra"
features = ["calls", "minutes", "messages", "mb_used"]


# In[10]:


sns.pairplot(df_user_behavior, hue=target, height=3)


# ## Splitting Data

# In[11]:


df_user_behavior_train, df_user_behavior_valid = df_user_behavior[features],df_user_behavior[target]


# In[12]:


features_train, features_test, target_train, target_test = train_test_split(
df_user_behavior_train,
df_user_behavior_valid,
test_size=0.2, 
random_state=random_state,
stratify=df_user_behavior_valid)


# In[13]:


display(features_train.info())


# In[14]:


display(features_test.info())


# In[15]:


display(target_train)


# In[16]:


display(target_test)


# In[17]:


features_train, features_valid, target_train, target_valid = train_test_split(
features_train,
target_train,
test_size=0.25, 
random_state=random_state)


# In[18]:


display(features_train.shape)


# In[19]:


display(features_train.info())


# In[20]:


display(target_train.shape)


# In[21]:


display(target_train)


# In[22]:


display(features_valid.shape)


# In[23]:


display(features_valid.info())


# In[24]:


display(target_valid.shape)


# In[25]:


display(target_valid)


# ## Testing Models

# 

# ### Decision Tree Classifier

# #### Training

# In[26]:


decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(features_train, target_train)


# In[27]:


display('First training score for Decision Tree Classifier:')
display(decision_tree_model.score(features_train, target_train))


# #### Testing

# In[28]:


dt_target_pred = decision_tree_model.predict(features_test)
dt_target_valid = decision_tree_model.predict(features_valid)


# In[29]:


display('First test score for Decision Tree Classifier:')
display(accuracy_score(target_test,dt_target_pred))


# #### Tuning Hyperparameters

# In[30]:


parameters = {
"random_state":[54321],
"max_depth":[*range(1,11,1)],
"min_samples_split":[*range(1,11,1)],
"min_samples_leaf":[*range(1,11,1)]}


# In[31]:


import warnings
warnings.filterwarnings("ignore")
decision_tree_model_tuned = GridSearchCV(DecisionTreeClassifier(), parameters, cv=5).fit(features_train, target_train)


# In[32]:


dt_tuned_target_pred = decision_tree_model_tuned.predict(features_test)


# In[33]:


display(decision_tree_model_tuned.score(features_train, target_train))


# In[34]:


display(accuracy_score(target_test, dt_tuned_target_pred))


# #### Validation

# In[35]:


best_model = DecisionTreeClassifier(**decision_tree_model_tuned.best_params_).fit(features_train, target_train)


# In[36]:


best_model_valid = best_model.predict(features_valid)


# In[37]:


display(best_model)


# In[38]:


display('Best model training score:')
display(best_model.score(features_train, target_train))


# In[39]:


display('Best model validation score:')
display(accuracy_score(target_valid, best_model_valid))


# <div class="alert alert-warning">
# <b>Reviewer's comment</b>
# 
# It would be really nice to clarify in the printed line what metric you're using and whether it is a train set or validation set score
# 
# </div>

# <div class="alert alert-warning">
# <b>Reviewer's comment</b>
# 
# Here it would be again nice to clarify which scores these are
# 
# </div>

# ### Random Forest Model

# #### Training

# In[40]:


random_forest_model = RandomForestClassifier()
random_forest_model.fit(features_train, target_train)


# In[41]:


display('First training score for Random Forest Classifier:')
display(random_forest_model.score(features_train, target_train))


# #### Testing

# In[42]:


dt_forest_target_pred = random_forest_model.predict(features_valid)


# In[43]:


display('First valid score for Decision Tree Classifier:')
display(accuracy_score(target_valid,dt_forest_target_pred))


# #### Tuning Hyperparameters

# In[44]:


parameters = {
"random_state":[54321],
"n_estimators":[*range(1,51,1)],
"max_depth":[*range(1,51,1)]}


# In[45]:


random_forest_model_tuned = GridSearchCV(RandomForestClassifier(), parameters, cv=5).fit(features_train, target_train)


# In[46]:


dt_tuned_forest_target_pred = random_forest_model_tuned.predict(features_valid)


# In[49]:


display('Tuned up Random Forest Classifier Score')
display(random_forest_model_tuned.score(features_train, target_train))


# In[50]:


display(accuracy_score(target_valid, dt_tuned_forest_target_pred))


# #### Final Test

# In[53]:


best_forest_model = RandomForestClassifier(**random_forest_model_tuned.best_params_).fit(features_train, target_train)


# In[54]:


best_forest_model_valid = best_forest_model.predict(features_test)


# In[55]:


display(best_forest_model)


# In[57]:


display('Best model training score:')
display(best_forest_model.score(features_train, target_train))


# In[58]:


display('Best model testing score:')
display(accuracy_score(target_valid, best_forest_model_valid))


# ### Logistics Regression

# #### Training

# In[59]:


logistics_model =  LogisticRegression()
logistics_model.fit(features_train,target_train)
score_train = logistics_model.score(features_train,target_train) 


# In[60]:


display("Score of the logistic regression model on the training set:",score_train)


# #### Testing

# In[61]:


logistics_prediction = logistics_model.predict(features_test) 
score_test = accuracy_score(target_test,logistics_prediction) 


# In[62]:


display("Accuracy of the logistic regression model on the test set:",score_test)


# In[63]:


logistics_model_test =  LogisticRegression()
logistics_model_test.fit(features_train,target_train)
logistics_valid = logistics_model.predict(features_valid)
score_valid = accuracy_score(target_valid, logistics_valid)  
score_test_valid = logistics_model_test.score(features_valid,target_valid)  


# In[64]:


display("Accuracy of the logistic regression model on the validation set:",score_valid)


# In[65]:


display("Score of the logistic regression model on the validation set:",score_test_valid)


# #### Tuning Hyperparameters

# In[66]:


logistics_tuned = LogisticRegression(verbose=0, random_state=random_state).fit(features_train, target_train)
valid_pred = logistics_tuned.predict(features_valid)


# In[67]:


display('Accuracy score for the final valid set.')
display(accuracy_score(target_valid,valid_pred))


# ### Sanity Check

# #### Never Ultra

# In[68]:


dummy_model = DummyClassifier(strategy='stratified').fit(features_train, target_train)
dummy_preds = dummy_model.predict(features_test)
display(f"Mean accuracy on test set: {dummy_model.score(features_test, target_test):.2f}")


# Random guessing gives accuracy close to 60%. Lower than all other ML models tested. However, the Random Forest Model did not score so well on its final test. It only score a 60%, and since about 70% of all the plans are not ultra, a model that just assumed Not Ultra would score higher than the random forest model. Likewise, the Logistics Regression only scored 69% on its test which is the same accuracy as assuming all plans are not ultra. Neither of those models were very effective. The Decision Tree Classifier on the other hand scored over 8
