from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import keras.backend as K
from optuna.samplers import TPESampler
import numpy as np
import keras
import optuna

class Objective(object):
    def __init__(self, dir_save, train_path, test_path, valid_path,
                 max_epochs, learn_rate_epochs):
        self.dir_save = dir_save
        self.train_path = train_path
        self.test_path = test_path
        self.valid_path = valid_path
        self.max_epochs = max_epochs
        self.learn_rate_epochs = learn_rate_epochs

    def __call__(self, trial):
        num_dense_nodes = trial.suggest_categorical('num_dense_nodes', [3000, 2500, 2000, 1500, 1000, 500])
        batch_size = trial.suggest_categorical('batch_size', [1, 5, 10, 32, 64])
        epochs_ = trial.suggest_categorical('num_epochs', [5, 10, 15, 20, 30, 40, 50])
        learning_rate_ = trial.suggest_categorical('num_epochs', [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005])
        drop_out = trial.suggest_discrete_uniform('drop_out', 0.05, 0.07, 0.09, 0.12, 0.15, 0.17, 0.19, 0.21)

        # clear session to boost optimization speed
        K.clear_session()

        # specify hyperparameters in dictionary
        dict_params = {'num_dense_nodes': num_dense_nodes,
                       'batch_size': batch_size,
                       'epochs': epochs_,
                       'drop_out': drop_out,
                       'learning_rate': learning_rate_}

        # pretrained convolutional layers are loaded using the Imagenet weights.
        # Include_top is set to False, in order to exclude the model's fully-connected layers.
        vgg16_model = keras.applications.VGG16(
            include_top=True,
            weights="imagenet",
            input_tensor=None,
            input_shape=(224, 224, 3),
            pooling=None,
            classes=2,
            classifier_activation="softmax")

        # delete last layer, and set the number of non-trainable parameters
        model = Sequential()
        for layer in vgg16_model.layers[:-1]:
            model.add(layer)
        for layer in model.layers:
            layer.trainable = False

        # add dense layers where the final dense layer has 2 units (e.g., binary classification)
        model.add(Dense(units=4096, activation='softmax'))
        model.add(Dropout(dict_params['drop_out']))
        model.add(Dense(units=dict_params['num_dense_nodes'], activation='softmax'))
        model.add(Dropout(dict_params['drop_out']))
        model.add(Dense(units=2, activation='softmax'))

        # compile the model
        clf_model = model.compile(optimizer=Adam(learning_rate=dict_params['learning_rate']),
                                  loss='categorical_crossentropy',
                                  metrics=['accuracy'])

        BATCH_SIZE = dict_params['batch_size']
        EPOCHS = dict_params['epochs']

        # get the images from the train and valid sets
        train_batches = ImageDataGenerator(preprocessing_function=keras.applications.vgg16.preprocess_input) \
            .flow_from_directory(directory=self.train_path, target_size=(224, 224), classes=['abnormal', 'normal'],
                                 batch_size=BATCH_SIZE)
        valid_batches = ImageDataGenerator(preprocessing_function=keras.applications.vgg16.preprocess_input) \
            .flow_from_directory(directory=self.valid_path, target_size=(224, 224), classes=['abnormal', 'normal'],
                                 batch_size=BATCH_SIZE)

        n_steps = train_batches.samples // BATCH_SIZE
        n_val_steps = valid_batches.samples // BATCH_SIZE

        # callbacks for early stopping and for learning rate reducer
        fn = self.dir_save + str(trial.number) + '_vgg16.h5'
        callbacks_list = [EarlyStopping(monitor='val_loss', patience=10),
                          ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                            patience=self.learn_rate_epochs,
                                            verbose=0, mode='auto', min_lr=1.0e-6),
                          ModelCheckpoint(filepath=fn,
                                          monitor='val_loss', save_best_only=True)]

        # fit the model
        history = clf_model.fit(x=train_batches,
                                batch_size=BATCH_SIZE,
                                epochs=EPOCHS,
                                validation_data=valid_batches,
                                steps_per_epoch=n_steps,
                                validation_steps=n_val_steps,
                                shuffle=True, verbose=1,
                                callbacks=callbacks_list)

        return history


if __name__ == '__main__':
    # specify specific settings
    maximum_epochs = 50
    early_stop_epochs = 10
    learning_rate_epochs = 5
    optimizer_direction = 'minimize'
    number_of_random_points = 25
    maximum_time = 60 * 60 * 6  # seconds
    results_directory = '/'
    train_path = '/train/'
    valid_path = '/valid/'
    test_path = '/test/'

    # define objective
    objective = Objective(results_directory, train_path, test_path, valid_path, maximum_epochs, learning_rate_epochs)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction=optimizer_direction,
                                sampler=TPESampler(n_startup_trials=number_of_random_points))

    # optimize hyperparameters
    study.optimize(objective, timeout=maximum_time)
