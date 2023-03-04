from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import pandas as pd
import numpy as np
import pickle
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

le = preprocessing.LabelEncoder()
le.fit(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'i', 'l', 'm', 'n', 'o', 'p',
        'q', 'r', 's', 't', 'u', 'v', 'w', 'y'])

dataset = pd.read_csv('dataset.csv', header=0)

X = np.array(dataset.iloc[:,:63])
Y = np.array(le.transform(dataset['classe']))

neigh = KNeighborsClassifier(n_neighbors=7)
neigh.fit(X, Y)

initial_type = [ 
    ( 'input_landmarks' , FloatTensorType( [None,63] ) ) 
]

converted_model = convert_sklearn(neigh , initial_types=initial_type )
with open( "alphabet_model.onnx", "wb" ) as f:
        f.write( converted_model.SerializeToString() )

pickle.dump(neigh, open('alphabet_model.pickle', "wb"))