import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 8)
plt.rcParams['legend.loc'] = "best"

SMALL_SIZE = 10
MEDIUM_SIZE = 13
BIGGER_SIZE = 16

plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)  # font-size of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # font-size of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)  # font-size of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)  # font-size of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend font-size
plt.rc('figure', titlesize=BIGGER_SIZE)  # font-size of the figure title

def plot_roc_curve(y_preds,
                   y_probas,
                   p_values,
                   model_names,
                   y_true,
                   title):
    """
       Plot the Receiver Operating Curves (ROC) from the trained binary classifiers.

       Parameters
       ----------
       y_preds : numpy array / list
            Multidimensional array of the predicted boolean classes, where (rows = models, columns = values (0,1)).

       y_probas : numpy array / list
            Multidimensional array of the predicted classifier probabilities, where (rows = models, columns = values between 0 and 1).

       p_values : numpy array/ list
            Multidimensional array of the p values from McNemar's test, where (rows = models, columns = p values).

       model_names : numpy array / list
            1D array with the model names.

       model_names : numpy array / list
            1D array with the correct boolean classes.

       title : str
            title name for the ROC plot

       Returns
       -------
            None
       """

    assert len(y_preds) == len(y_probas)
    assert y_preds.shape[1] == len(model_names)

    tprs = []
    fprs = []
    for i in range(len(y_preds)):
        # make false and true positive rates
        fpr, tpr, _ = metrics.roc_curve(y_preds[i], y_probas[i])
        auc = metrics.roc_auc_score(y_preds[i], y_probas[i])
        tprs.append(tpr)
        fprs.append(fpr)

        # calculate the specificity and sensitivity
        tn, fp, fn, tp = confusion_matrix(y_preds[i], y_true).ravel()
        spec = tn / (tn + fp)
        sens = tp / (tp + fn)

        print(model_names[i])
        print(spec, sens)
        print()

        # plot the ROC curve of the model
        line_type = ['o-', 'v-', '^-', '<-', '>-', '8-', 's-', 'p-', '*-', 'h-', 'H-', 'D-', 'd-', 'P-', 'X-']
        plt.plot(fpr, tpr, line_type[i],
                 label="{}: auc={}, p={}".format(model_names[i], round(auc, 3), round(p_values[i], 3)),
                 linewidth=1, markersize=5)
        plt.legend(loc=4)
        plt.title(model_names[i])

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()
    
