from sklearn.preprocessing import MinMaxScaler,Normalizer,StandardScaler,OneHotEncoder,LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import GridSearchCV
from pandas import set_option
from numpy import set_printoptions
from sklearn.metrics import mean_squared_error,accuracy_score, r2_score
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression,Ridge,RidgeClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.ensemble import BaggingRegressor,RandomForestRegressor,GradientBoostingRegressor
import urllib
from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import KFold,cross_val_score,train_test_split
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.linear_model import SGDClassifier,PassiveAggressiveClassifier
from sklearn.neighbors import NearestCentroid
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from mlxtend.evaluate import bias_variance_decomp

#loading data
df = pd.read_csv('../housing.csv').fillna(0)



X = df.iloc[:,:9]

Y = df.iloc[:,9]
#label encode
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(Y)

#scaling data
scaler = StandardScaler()
X = scaler.fit_transform(X)
#slicing data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2,shuffle=True, random_state=7)
#bias variance

avg_loss,avg_bias,avg_var = bias_variance_decomp(DecisionTreeClassifier(),
                                                    X_train,
                                                    Y_train,
                                                    X_test,
                                                    Y_test,
                                                    loss='0-1_loss',
                                                    num_rounds=20)

print(avg_bias, avg_var, avg_loss) # 0.01986434108527132 0.02508478682170543 0.03238856589147287

