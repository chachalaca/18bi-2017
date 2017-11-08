
# coding: utf-8

# In[165]:

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import datetime
import time

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

import statsmodels.api as sm
import statsmodels.formula.api as smf

from fbprophet import Prophet

get_ipython().magic('matplotlib inline')


# In[56]:

df = pd.read_table('wisconsin-employment-time-series.tsv', names=['month', 'value'], skiprows=1, dtype={"value": np.float16}).dropna()



# In[57]:

df.plot(x=["month"], y=["value"])


# In[62]:

df["y"] = df.month.apply(lambda x: int(x.split("-",1)[0]))
df["m"] = df.month.apply(lambda x: int(x.split("-",1)[1]))


# In[89]:

df.plot(x="m", y="value")


# In[90]:

df.plot(x="y", y="value")


# ## Scikit-learn linear regression

# In[127]:

X = df[["m", "y"]]
y = df["value"]


# In[159]:

enc = OneHotEncoder()
XS = enc.fit_transform(pd.DataFrame({"m": X["m"]}))
XX = pd.DataFrame(XS.toarray(), dtype=np.int)
XX[["m", "y"]] = df[["m", "y"]]


# In[161]:

regr = linear_model.LinearRegression()
regr.fit(XX[:85], y[:85])
y_pred = regr.predict(XX[85:])


# In[162]:

res = pd.DataFrame({"true": y[85:], "predicted": y_pred})
res.plot(y=["predicted", "true"])


# In[164]:

print('Coefficients: \n', regr.coef_)
print("Mean squared error: %.2f"
      % mean_squared_error(res.true, res.predicted))
print('Variance score: %.2f' % r2_score(res.true, res.predicted))


# ## Facebook Prophet

# In[167]:

df["day"] = df.month.apply(lambda x: x+"-01")


# In[175]:

m = Prophet()
m.fit(pd.DataFrame({"ds": df["day"][:85], "y": y[:85]}))


# In[180]:

forecast = m.predict(pd.DataFrame({"ds": df["day"][85:]}))
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[197]:

res2 = pd.DataFrame({"day": df["day"][85:].reset_index().day, "true": y[85:].reset_index().value, "predicted": forecast["yhat"]})
res2.plot(y=["predicted", "true"], x="day")


# In[198]:

print("Mean squared error: %.2f"
      % mean_squared_error(res2.true, res2.predicted))
print('Variance score: %.2f' % r2_score(res2.true, res2.predicted))


# In[199]:

m.plot(forecast)


# In[200]:

m.plot_components(forecast)

