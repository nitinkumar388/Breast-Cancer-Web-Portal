import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.datasets import load_breast_cancer


if __name__ == '__main__':
    
    cancer = load_breast_cancer()
    data = pd.DataFrame(cancer.data)
    data.columns = cancer.feature_names

    df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns=np.append(cancer['feature_names'], ['target']))

    X = df_cancer.drop(['target'], axis=1)
    Y = df_cancer['target']

    x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=5)

    
    min_train = x_train.min()
    range_train = (x_train-min_train).max()
    x_train_scaled = (x_train-min_train)/range_train

    min_test = x_test.min()
    range_test = (x_test-min_test).max()
    x_test_scaled = (x_test-min_test)/range_test



    clf = SVC(probability=True)

    clf.fit(x_train_scaled, y_train)
    y_pred1 = clf.predict(x_test_scaled)



    #clf.fit(x_train, y_train)

    #y_pred = clf.predict(x_test)

    #min_train = x_train.min()
    #range_train = (x_train-min_train).max()
    #x_train_scaled = (x_train-min_train)/range_train

    #min_test = x_test.min()
    #range_test = (x_test-min_test).max()
    #x_test_scaled = (x_test-min_test)/range_test

    #clf.fit(x_train_scaled, y_train)
    #y_pred1 = clf.predict(x_test_scaled)


    #inputFeatures = [0.025,0.26,0.145,0.15,0.215,0.15,0.6887,0.15,0.0216,0.156,0.36,0.15,0.15,0.1,0.3974,0.15,0.28,0.34,0.15,0.148,0.156,0.975,0.165,0.15,0.852,0.96,0.78,0.354,0.785,0.12]
    #infProb = svc_model.predict_proba([inputFeatures])[0][1]
    #print(infProb)

    

    #param_grid = {'C':[0.1,1,10,100],'gamma':[1,0.1,0.01,0.001],'kernel':['rbf']}
    #grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=4)
    #grid.fit(x_train_scaled, y_train)
    #grid.best_params_
    #grid.best_estimator_
    #grid_predictions = grid.predict(x_test_scaled)


   # inputFeatures = [0.025,0.26,0.145,0.15,0.215,0.15,0.6887,0.15,0.0216,0.156,0.36,0.15,0.15,0.1,0.3974,0.15,0.28,0.34,0.15,0.148,0.156,0.975,0.165,0.15,0.852,0.96,0.78,0.354,0.785,0.12]
   # infProb = grid.predict([inputFeatures])[0]
   # print(infProb)


    file = open('model.pkl','wb')
    pickle.dump(clf,file)
    file.close()
    