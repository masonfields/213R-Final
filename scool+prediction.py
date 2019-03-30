
# coding: utf-8

# In[20]:


import pandas as pd
import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.model_selection import RandomizedSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import f1_score, classification_report, accuracy_score

get_ipython().magic('matplotlib inline')


# In[21]:


scores=pd.read_csv("C:\\Users\\Mason\\Desktop\\port.csv", sep=';' )
scores.head(10)


# In[22]:


target=scores.G3


# In[23]:


data=scores


# In[24]:


data['school']=pd.get_dummies(data.school)
data['sex']=pd.get_dummies(data.sex)
data['address']=pd.get_dummies(data.address)
data['famsize']=pd.get_dummies(data.famsize)
data['Pstatus']=pd.get_dummies(data.Pstatus)
data['schoolsup']=pd.get_dummies(data.schoolsup)
data['famsup']=pd.get_dummies(data.famsup)
data['paid']=pd.get_dummies(data.paid)
data['activities']=pd.get_dummies(data.activities)
data['nursery']=pd.get_dummies(data.nursery)
data['higher']=pd.get_dummies(data.higher)
data['internet']=pd.get_dummies(data.internet)
data['romantic']=pd.get_dummies(data.romantic)


# In[25]:


data['parentedu']=data.Medu+data.Fedu


# In[26]:


sns.distplot(data.parentedu)


# In[27]:


sns.lmplot(x="parentedu", y='G3', data=scores)


# In[29]:


sns.lmplot(x="higher", y='G3', data=scores)


# In[28]:


sns.lmplot(x="paid", y='G3', data=scores)


# In[10]:


sns.lmplot(x="studytime", y='G3', data=scores)


# In[31]:


data["social"]=data.goout+data.Dalc+data.Walc+data.freetime


# In[32]:


sns.lmplot(x="social", y='G3', data=data)


# In[30]:


sns.lmplot(x="freetime", y='G3', data=scores)


# In[14]:


for_corr=data
data=scores.drop(['G1','G2', 'G3'], axis=1)
data.isnull().sum()


# In[33]:


numbers=data.select_dtypes(include=[np.number])


# In[34]:


scale=StandardScaler()
scale.fit(numbers)
scale_df=scale.transform(numbers)


# In[35]:


X_train, X_test, y_train, y_test = train_test_split(scale_df,target, random_state=42, test_size=.33)


# In[36]:


lr=linear_model.LinearRegression()
model=lr.fit(X_train, y_train)
predictions=model.predict(X_test)
print('RMSE is: \n', mean_squared_error(y_test, predictions))


# In[37]:


sns.distplot(target, kde=False)
plt.title('Target Distribution')


# In[38]:


param_dist = {"eta0": [.001, .003, .01, .03, .1, .3, 1, 3], "tol": [.01, .001, .0001]}
n_iter_search= 8 
model=SGDRegressor()
random_search=RandomizedSearchCV(model, param_distributions=param_dist,
                                n_iter=n_iter_search, cv=3, scoring='neg_mean_squared_error')
random_search.fit(X_train, y_train)

print("Best Parameters: {}".format(random_search.best_params_))


# In[39]:


linear_regression_model = SGDRegressor(tol=.0001, eta0=.1)
linear_regression_model.fit(X_train, y_train)
train_predictions = linear_regression_model.predict(X_train)
test_predictions = linear_regression_model.predict(X_test)

train_mse = mean_squared_error(y_train, train_predictions)
test_mse = mean_squared_error(y_test, test_predictions)

print("Train RMSE: {}".format(np.sqrt(train_mse)))
print("Test RMSE: {}".format(np.sqrt(test_mse)))


# In[42]:


from sklearn.utils import resample
n_bootstraps = 1000
bootstrap_X = []
bootstrap_y = []
for _ in range(n_bootstraps):
    sample_X, sample_y = resample(scale_df, target)
    bootstrap_X.append(sample_X)
    bootstrap_y.append(sample_y)


# In[43]:


linear_regression_model = SGDRegressor(tol=.0001, eta0=.01)
coeffs = []
for i, data in enumerate(bootstrap_X):
    linear_regression_model.fit(data, bootstrap_y[i])
    coeffs.append(linear_regression_model.coef_)


# In[47]:


data_df=pd.DataFrame(data)
coef_df = pd.DataFrame(coeffs, columns=data_df.columns)
coef_df.plot(kind='box')
plt.xticks(rotation=90)


# In[48]:



corr=for_corr.corr()
corr.sort_values(["G3"], ascending=False, inplace=True)
print(corr.G3)


# In[52]:


coef_df[28].plot(kind='hist')


# In[ ]:


target.shape


# In[53]:


from sklearn.linear_model import ElasticNetCV



clf = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], alphas=[.1, 1, 10])
model=clf.fit(X_train, y_train)
train_predictions = clf.predict(X_train)
test_predictions = clf.predict(X_test)
train_mse=mean_squared_error(y_train, train_predictions)
test_mse=mean_squared_error(y_test, test_predictions)
print("Train RMSE: {}".format(np.sqrt(train_mse)))
print("Test RMSE: {}".format(np.sqrt(test_mse)))


# In[54]:


from sklearn.model_selection import learning_curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='neg_mean_squared_error')
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# In[55]:


plot_learning_curve(model, "Learning Curve", X_train, y_train, cv=5)


# In[56]:


from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, classification_report, accuracy_score, mean_squared_error


# In[60]:


reg = KNeighborsRegressor()
gridsearch = GridSearchCV(reg, {"n_neighbors": [1, 3, 5, 7, 9, 11], "weights": ['uniform', 'distance'], 
                                'p': [1, 2, 3]}, scoring='neg_mean_squared_error')
grid=gridsearch.fit(X_train, y_train)
print("Best Params: {}".format(gridsearch.best_params_))
y_pred_train = gridsearch.predict(X_train)
y_pred_test = gridsearch.predict(X_test)
print("Train MSE: {}\tTest MSE: {}".format(mean_squared_error(y_train, y_pred_train),
                                           mean_squared_error(y_test, y_pred_test)))


# In[61]:


plot_learning_curve(grid, "Learning Curve", X_train, y_train, cv=5)


# In[64]:


from sklearn import svm


# In[85]:



clf = svm.SVR(C=1)
clf.fit(X_train, y_train)

train_predictions = clf.predict(X_train)
test_predictions = clf.predict(X_test)
train_mse=mean_squared_error(y_train, train_predictions)
test_mse=mean_squared_error(y_test, test_predictions)
print("Train RMSE: {}".format(np.sqrt(train_mse)))
print("Test RMSE: {}".format(np.sqrt(test_mse)))

