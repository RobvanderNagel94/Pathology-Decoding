from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.utils import shuffle
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle

# ignore ConvergenceWarnings
import warnings
warnings.filterwarnings('ignore')

random_state = 42

meta = [
    {"estimator": LogisticRegression(),
     "params": {
         "solver": ['newton-cg', 'lbfgs', 'liblinear'],
         "penalty": ['none', 'l1', 'l2', 'elasticnet'],
         "C": [1000, 100, 10, 1.0, 0.1, 0.01, 0.001, 0.0001]
     }
     },
    {"estimator": SVC(),
     "params": {
         "C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
         "gamma": [100, 10, 1, 0.1, 0.01, 0.001, 0.0001],
         "kernel": ['rbf']
     }
     },
    {"estimator": RandomForestClassifier(),
     "params": {
         "criterion": ['gini', 'entropy'],
         "bootstrap": [True, False],
         "max_depth": [10, 20, 30, 40, 50, 60, 80],
         "max_features": ['sqrt', 'log2'],
         "n_estimators": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 200, 250, 300, 350],
         "random_state": [random_state]
     }
     },
    {"estimator": KNeighborsClassifier(),
     "params": {
         "n_neighbors": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80],
         "weights": ['uniform'],
         "leaf_size": [20, 25, 30, 35, 40, 45, 50],
         "p": [1, 2, 3]
     }
     },
    {"estimator": MLPClassifier(),
     "params": {
         "hidden_layer_sizes": [(10, 30, 10), (20,)],
         "activation": ['tanh', 'relu'],
         "solver": ['lbfgs', 'sgd', 'adam'],
         "alpha": [0.0005, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1],
         "learning_rate": ['adaptive'],
         "random_state": [random_state]
     }
     },
]

def classifier_param_search(meta,
                            df,
                            sample_size=431,
                            test_size=80,
                            random_state=42):
    """
       Iterative feature selection procedure to find the most optimal set of computational features, and
       saves the best parameters to pickle file.

       Parameters
       ----------
       meta : gridsearch parameters
            parameter definitions for the different binary classifiers.

       df : pandas dataframe
            dataset with labels and computational features (first column are the labels).

       sample_size : int
            size to increase the initial minority class to get a balanced dataset (bootstrapping with replacement).

       test_size : int
            size for test set (test_size*2 samples).

       random_state : int
            set seed for reproducibility.

       Returns
       -------
            None
       """

    # shuffle dataset
    dataset = df.copy()
    dataset = shuffle(dataset, random_state=random_state)

    # construct test set for normal/abnormal
    dataset_ab_test = dataset[dataset.label_bool == 1][:test_size]
    dataset_no_test = dataset[dataset.label_bool == 0][:test_size]
    dataset_ab_train = dataset[dataset.label_bool == 1][test_size:]
    dataset_no_train = dataset[dataset.label_bool == 0][test_size:]

    # construct train and test sets
    dataset_train = pd.concat([dataset_ab_train, dataset_no_train], axis=0)
    dataset_test = pd.concat([dataset_ab_test, dataset_no_test], axis=0)

    y_train = dataset_train.label_bool.values
    X_train = dataset_train.iloc[:, 1:].values
    y_test = dataset_test.label_bool.values
    X_test = dataset_test.iloc[:, 1:].values

    # Scale values for train and test features
    sc = MinMaxScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)

    accuracies = []
    params = []
    for model in tqdm(meta):
        clf = model["estimator"]
        print(clf)

        # define the grid search
        grid_search = GridSearchCV(estimator=clf, param_grid=model["params"], cv=8)
        # perform the search
        grid_search.fit(X_train_scaled, y_train)

        # make a prediction on the test set
        y_pred = grid_search.predict(X_test_scaled)

        # compute the model's accuracy
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        acc = (tp + tn) / (tp + fp + fn + tn)
        accuracies.append(acc)
        params.append(grid_search.best_params_)

        # print best parameters
        print(grid_search.best_params_)
        print()

    # save parameters to pickle
    with open('best_params.pkl', 'wb') as f:
        pickle.dump(params, f)
    with open('best_acc.pkl', 'wb') as f:
        pickle.dump(accuracies, f)
