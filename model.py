import quandl as qd
import numpy as np
import pandas as pd
import datetime as dt
import pickle
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

qd.ApiConfig.api_key = API_KEY
df = qd.get('WIKI/GOOGL', start_date=dt.date(2013, 1, 1))

df = pd.DataFrame(df, columns=['Adj. Close'])
df = df.reset_index()

fd = 500
df['Prediction'] = df[['Adj. Close']].shift(-fd)

X = np.array(df.drop(['Prediction','Date'],1))[:-fd]
y = np.array(df['Prediction'])[:-fd].reshape(-1,1)
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)

xfd = (df.drop(['Date','Prediction'],1))[:-fd].tail(fd) 
xfd = np.array(xfd)

regressor=LinearRegression()
regressor.fit(xtrain,ytrain)

pickle.dump(regressor, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[25]]))
