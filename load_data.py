import pandas as pd
import numpy as np
# visualization
import matplotlib.pyplot as plt
pd.options.display.mpl_style = 'default'

# load data pandas dataframe
key = pd.io.parsers.read_csv('key.csv')
train = pd.io.parsers.read_csv('train.csv')
test = pd.io.parsers.read_csv('test.csv')
weather = pd.io.parsers.read_csv('weather.csv')
# Cleanup weather data
weather = weather.replace('M','-',regex=True)
weather = weather.replace(' ','-',regex=False)
weather = weather.replace('-',np.nan,regex=False)
# Replace trace values with numerical values

# join key and weather - get store and station nbrs
kw_join = pd.merge(key,weather, left_on='station_nbr',right_on='station_nbr',how='inner')

# run baseline classifier
from sklearn import svm

reg = svm.LinearSVC(verbose=1)
X = np.array(train.item_nbr)
X = X.reshape((X.shape[0],1))
reg.fit(X, np.array(train.units))
result = reg.predict(np.array(train.item_nbr))

print result

# run clustering alg
from sklearn import cluster

km = cluster.KMeans(n_clusters=10,init='k-means++',n_init=10, max_iter=300,

# write result to file