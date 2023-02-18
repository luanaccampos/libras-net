from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import pickle

dataset = pd.read_csv('dataset.csv', header=0)

X = np.array(dataset.iloc[:,:63])
Y = np.array(dataset['classe'])

neigh = KNeighborsClassifier(n_neighbors=7)
neigh.fit(X, Y)

pickle.dump(neigh, open('alphabet_model.pickle', "wb"))