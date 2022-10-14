from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
import pandas as pd
import numpy as np

def select_k_best_features(clf,
                           df,
                           sample_size=431,
                           test_size=80,
                           random_state=42,
                           method=f_classif):
    """
       Iterative feature selection procedure to find the most optimal set of computational features.

       Parameters
       ----------
       clf : classifier 
            binary classifier (e.g., Logistic Regression, Random Forest, Support Vector Machine, K-nearest Neighbors,
            Multi-layer perceptron).

       df : pandas dataframe
            dataset with computational labels and features. (first column are the labels)

       sample_size : int
            size to increase the initial minority class to get a balanced dataset (bootstrapping with replacement).

       test_size : int
            size for test set (test_size*2 samples).

       random_state : int
            set seed for reproducibility.

       method : feature selection function
            method to select features (f-test or mutual information).

       Returns
       -------

       sorted_feature_importance : dict
            sorted dictionary of relevant features and their importance (descending order)

       """

    def select_features(X_train, y_train, X_test, k, method):
        fs = SelectKBest(score_func=method, k=k)
        fs.fit(X_train, y_train)
        X_train_fs = fs.transform(X_train)
        X_test_fs = fs.transform(X_test)
        return X_train_fs, X_test_fs, fs

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
    X_test = dataset_test.iloc[:, 1:].values

    # define a pipeline construction
    fs = SelectKBest(score_func=method)
    pipeline = Pipeline(steps=[('anova', fs), ('clf', clf)])

    # define the grid
    grid = dict()
    grid['anova__k'] = [i + 1 for i in range(X_train.shape[1])]

    # define the grid search
    grid_search = GridSearchCV(pipeline, grid, scoring='accuracy', n_jobs=-1, cv=8)
    # perform the search
    grid_search.fit(X_train, y_train)

    # summarize best features
    print('Best Mean Accuracy: %.3f' % grid_search.best_score_)
    print('Best Config: %s' % grid_search.best_params_)

    # return dict with the optimal amount of features to select
    k = grid_search.best_params_['anova__k']

    # apply feature selection
    X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test, k, method)
    feature_names = dataset_train.iloc[:, 1:].columns

    # compute the feature importance's
    dic = dict()
    for i in range(len(fs.scores_)):
        print('Feature %d: %f' % (i, fs.scores_[i]))
        dic[feature_names[i]] = fs.scores_[i]

    # sort features in descending order
    sorted_feature_importance = sorted(dic.items(), key=lambda x: x[1], reverse=True)
    return sorted_feature_importance
