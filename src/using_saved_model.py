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



loaded_model = pickle.load(open('final_model.sav', 'rb'))
#load_data
df = pd.read_csv('../housing.csv').fillna(0)



X = df.iloc[:,:9]

Y = df.iloc[:,9]
#label encode
label_encoder = LabelEncoder()


#scaling data
scaler = StandardScaler()

#slicing data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2,shuffle=True, random_state=7)
X_test = scaler.fit_transform(X_test)
Y_test = label_encoder.fit_transform(Y_test)
result = loaded_model.score(X_test, Y_test)
print(result) #93%
print()