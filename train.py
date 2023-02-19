from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import pandas as pd
import numpy as np
import pickle
from sklite import LazyExport

le = preprocessing.LabelEncoder()
le.fit(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'i', 'l', 'm', 'n', 'o', 'p',
        'q', 'r', 's', 't', 'u', 'v', 'w', 'y'])

dataset = pd.read_csv('dataset.csv', header=0)

X = np.array(dataset.iloc[:,:63])
Y = np.array(le.transform(dataset['classe']))

neigh = KNeighborsClassifier(n_neighbors=7)
neigh.fit(X, Y)

lazy = LazyExport(neigh)
lazy.save('alphabet_model.json')

pickle.dump(neigh, open('alphabet_model.pickle', "wb"))