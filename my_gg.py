import numpy as np
import keras
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=1000)

tokenizer = Tokenizer(num_words=1000)
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')




from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier(my_opti, nodes, drop):
    classifier = Sequential()
    
    classifier.add(Dense(units = nodes, input_dim = 1000, activation = 'relu')) 
    classifier.add(Dropout(drop))
    
    classifier.add(Dense(units = nodes, activation = 'relu'))
    classifier.add(Dropout(drop))
    
    classifier.add(Dense(units = nodes, activation = 'relu'))
    classifier.add(Dropout(drop))
    
    classifier.add(Dense(units = 2, activation = "softmax"))
    
    classifier.compile(optimizer = my_opti, loss = "categorical_crossentropy", metrics = ["accuracy"])
    
    return classifier 

classifier = KerasClassifier(build_fn = build_classifier, epochs = 500)
 
parameters = {'batch_size': [10, 32], 
              'epochs' : [10, 20],
              'nodes' : [512, 1024],
              'my_opti': ['adam', 'rmsprop','nadam'],
              'drop' : [0.3, 0.5]}

from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)

grid_search = grid_search.fit(x_train, y_train)
best_par = grid_search.best_params_
best_acc = grid_search.best_score_




























































