import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.figsize"] = (10, 8)

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

def plot_history(history, epochs):
    N = epochs
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), history.history["val_loss"], label="validation")
    plt.plot(np.arange(0, N), history.history["val_acc"], label="train")
    plt.title("Train and validation losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
