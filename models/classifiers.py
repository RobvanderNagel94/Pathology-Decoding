from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.utils import shuffle
import numpy as np
import pandas as pd

# ignore ConvergenceWarnings
import warnings
warnings.filterwarnings('ignore')

random_state = 42

# define binary classifiers and optimal parameters
clf_LR = LogisticRegression(solver='lbfgs', penalty='none', C=100)
clf_SVM = SVC(C=100, gamma=0.0001, kernel='rbf')
clf_RF = RandomForestClassifier(criterion='entropy', bootstrap=True, max_depth=30, max_features='sqrt', n_estimators=80,
                                random_state=random_state)
clf_KNN = KNeighborsClassifier(n_neighbors=40, weights='uniform', leaf_size=30, p=1)
clf_MLP = MLPClassifier(hidden_layer_sizes=(20,), activation='tanh', solver='sgd', alpha=0.0001,
                        learning_rate='adaptive', random_state=random_state)
clf_MVC3 = VotingClassifier(estimators=[('lr', clf_LR), ('rf', clf_RF), ('knn', clf_KNN)], voting='hard')

clfs = [clf_LR, clf_SVM, clf_RF, clf_KNN, clf_MLP, clf_MVC3]
model_names = ['LR', 'SVM', 'RF', 'KNN', 'MLP', 'MVC3']

def classifier_predict(clfs,
                       df,
                       sample_size=431,
                       test_size=80,
                       random_state=42):
    """
       Function to compute the classifier predictions.

       Parameters
       ----------
       clfs : list
            sci-kit learn classifiers with optimal parameter settings.

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
       y_preds : list
            Multi-list of the predicted boolean classes, where (rows = models, columns = values (0,1))

       y_probas :  list
            Multi-list of the predicted classifier probabilities, where (rows = models, columns = values between 0 and 1).

       accuracies :  list
            Multi-list of the model's accuracies, where (rows = models, columns = values between 0 and 100%).

       specificities : list
            Multi-list of the model's specificities, where (rows = models, columns = values between 0 and 100%).

       sensitivities : list
            Multi-list of the model's sensitivities, where (rows = models, columns = values between 0 and 100%).
       """

    # shuffle dataset
    dataset = df.copy()
    dataset = shuffle(dataset, random_state=random_state)

    # construct test set for normal/abnormal
    dataset_ab_test = dataset[dataset.label_bool == 1][:test_size]
    dataset_no_test = dataset[dataset.label_bool == 0][:test_size]
    dataset_ab_train = dataset[dataset.label_bool == 1][test_size:]
    dataset_no_train = dataset[dataset.label_bool == 0][test_size:]

    # bootstrap with resampling for the minority class
    class_ab = np.random.choice(dataset_ab_train.idx.values, size=sample_size)
    class_ab_df = pd.concat([dataset_ab_train[dataset_ab_train.idx == class_ab[i]] for i in range(len(class_ab))])
    df_class_ab_bootstrap = pd.concat([class_ab_df, dataset_ab_train])

    # construct train and test sets
    dataset_train = pd.concat([dataset_no_train, df_class_ab_bootstrap], axis=0)
    dataset_test = pd.concat([dataset_ab_test, dataset_no_test], axis=0)

    y_train = dataset_train.label_bool.values
    X_train = dataset_train.iloc[:, 1:].values
    y_test = dataset_test.label_bool.values
    X_test = dataset_test.iloc[:, 1:].values

    # scale values for train and test features
    sc = MinMaxScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)

    y_preds = []
    y_probas = []
    accuracies = []
    specificities = []
    sensitivities = []
    for clf in clfs:
        # fit classifier to train data
        model = clf.fit(X_train_scaled, y_train)

        # make prediction on the test set
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)

        # compute the model's accuracy
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        acc = (tp + tn) / (tp + fp + fn + tn)
        spec = tn / (tn + fp)
        sens = tp / (tp + fn)
        print(acc, spec, sens)

        # save model predictions
        y_preds.append(y_pred)
        y_probas.append(y_proba)
        accuracies.append(acc)
        specificities.append(spec)
        sensitivities.append(sens)

    # return classifier results
    return y_preds, y_probas, accuracies, specificities, sensitivities
