from braindecode.datautil.signal_target import SignalAndTarget
from braindecode.torch_ext.util import np_to_var
from braindecode.torch_ext.util import set_random_seeds
from braindecode.torch_ext.modules import Expression
from braindecode.experiments.experiment import Experiment
from braindecode.datautil.iterators import CropsFromTrialsIterator
from braindecode.experiments.monitors import (RuntimeMonitor, LossMonitor, MisclassMonitor)
from braindecode.experiments.stopcriteria import MaxEpochs
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.models.deep4 import Deep4Net
from braindecode.models.util import to_dense_prediction_model
from braindecode.datautil.iterators import get_balanced_batches
from braindecode.torch_ext.constraints import MaxNormDefaultConstraint
from braindecode.torch_ext.util import var_to_np
from braindecode.torch_ext.functions import identity
from braindecode.datautil.iterators import _compute_start_stop_block_inds
from numpy.random import RandomState
from copy import copy
from sklearn.metrics import roc_auc_score
from torch import optim
import torch.nn.functional as F
import torch as th
from torch.nn.functional import elu
from torch import nn
import re
import glob
import os.path
import mne
import sys
import numpy as np
import logging
import time
import resampy

log = logging.getLogger(__name__)
log.setLevel('DEBUG')

def get_all_sorted_file_names_and_labels(train_or_eval,
                                         path_train_abnormal,
                                         path_train_normal,
                                         path_eval_abnormal,
                                         path_eval_normal,
                                         ):
    if train_or_eval == 'train':
        os.chdir(path_train_normal)
        edf_files_no = np.sort([f for f in glob.glob('*.edf')])
        labels_no = np.array([0] * len(edf_files_no))
        os.chdir(path_train_abnormal)
        edf_files_ab = np.sort([f for f in glob.glob('*.edf')])
        labels_ab = np.array([1] * len(edf_files_ab))
        all_file_names = np.array([edf_files_no, edf_files_ab]).flatten()
        labels = np.array([labels_no, labels_ab]).flatten().astype(np.int64)
        print(all_file_names)
        print(labels)

    elif train_or_eval == 'eval':

        os.chdir(path_eval_normal)
        edf_files_no = np.sort([f for f in glob.glob('*.edf')])
        labels_no = np.array([0] * len(edf_files_no))
        os.chdir(path_eval_abnormal)
        edf_files_ab = np.sort([f for f in glob.glob('*.edf')])
        labels_ab = np.array([1] * len(edf_files_ab))
        all_file_names = np.array([edf_files_no, edf_files_ab]).flatten()
        labels = np.array([labels_no, labels_ab]).flatten().astype(np.int64)

    return all_file_names, labels

def get_recording_length(file_path):
    """ some recordings were that huge that simply opening them with mne caused the program to crash. therefore, open
    the edf as bytes and only read the header. parse the duration from there and check if the file can safely be opened
    :param file_path: path of the directory
    :return: the duration of the recording
    """
    f = open(file_path, 'rb')
    header = f.read(256)
    f.close()
    return int(header[236:244].decode('ascii'))

def session_key(file_name):
    """ sort the file name by session """
    return re.findall(r'(s\d{2})', file_name)

def create_set(X, y, inds):
    new_X = []
    for i in inds:
        new_X.append(X[i])
    new_y = y[inds]
    return SignalAndTarget(new_X, new_y)

def natural_key(file_name):
    """ provides a human-like sorting key of a string """
    key = [int(token) if token.isdigit() else None
           for token in re.split(r'(\d+)', file_name)]
    return key

def time_key(file_name):
    """ provides a time-based sorting key """
    splits = file_name.split('/')
    [date] = re.findall(r'(\d{4}_\d{2}_\d{2})', splits[-2])
    date_id = [int(token) for token in date.split('_')]
    recording_id = natural_key(splits[-1])
    session_id = session_key(splits[-2])

    return date_id + session_id + recording_id

def read_all_file_names(path, extension, key="time"):
    """ read all files with specified extension from given path
    :param path: parent directory holding the files directly or in subdirectories
    :param extension: the type of the file, e.g. '.txt' or '.edf'
    :param key: the sorting of the files. natural e.g. 1, 2, 12, 21 (machine 1, 12, 2, 21) or by time since this is
    important for cv. time is specified in the edf file names
    """
    file_paths = glob.glob(path + '**/*' + extension, recursive=True)

    if key == 'time':
        return sorted(file_paths, key=time_key)

    elif key == 'natural':
        return sorted(file_paths, key=natural_key)


def get_info_with_mne(file_path):
    """ read info from the edf file without loading the data. loading data is done in multiprocessing since it takes
    some time. getting info is done before because some files had corrupted headers or weird sampling frequencies
    that caused the multiprocessing workers to crash. therefore get and check e.g. sampling frequency and duration
    beforehand
    :param file_path: path of the recording file
    :return: file name, sampling frequency, number of samples, number of signals, signal names, duration of the rec
    """
    try:
        edf_file = mne.io.read_raw_edf(file_path, verbose='error')
    except ValueError:
        return None, None, None, None, None, None

    # some recordings have a very weird sampling frequency. check twice before skipping the file
    sampling_frequency = int(edf_file.info['sfreq'])
    if sampling_frequency < 10:
        sampling_frequency = 1 / (edf_file.times[1] - edf_file.times[0])
        if sampling_frequency < 10:
            return None, sampling_frequency, None, None, None, None

    n_samples = edf_file.n_times
    signal_names = edf_file.ch_names
    n_signals = len(signal_names)
    # some weird sampling frequencies are at 1 hz or below, which results in division by zero
    duration = n_samples / max(sampling_frequency, 1)

    return edf_file, sampling_frequency, n_samples, n_signals, signal_names, duration

def load_data(fname, preproc_functions, sensor_types=['EEG']):
    cnt, sfreq, n_samples, n_channels, chan_names, n_sec = get_info_with_mne(
        fname)
    log.info("Load data...")
    cnt.load_data()
    selected_ch_names = []
    if 'EEG' in sensor_types:
        wanted_elecs = ['Fp2', 'Fp1', 'F8', 'F7', 'F4', 'F3', 'T4', 'T3', 'C4', 'C3',
                        'T6', 'T5', 'P4', 'P3', 'O2', 'O1', 'Fz', 'Cz', 'Pz']
        for wanted_part in wanted_elecs:
            wanted_found_name = []
            for ch_name in cnt.ch_names:
                if ' ' + wanted_part + '-' in ch_name:
                    wanted_found_name.append(ch_name)
            assert len(wanted_found_name) == 1
            selected_ch_names.append(wanted_found_name[0])
    if 'EKG' in sensor_types:
        wanted_found_name = []
        for ch_name in cnt.ch_names:
            if 'EKG' in ch_name:
                wanted_found_name.append(ch_name)
        assert len(wanted_found_name) == 1
        selected_ch_names.append(wanted_found_name[0])

    cnt = cnt.pick_channels(selected_ch_names)

    assert np.array_equal(cnt.ch_names, selected_ch_names)
    n_sensors = 0
    if 'EEG' in sensor_types:
        n_sensors += 19
    if 'EKG' in sensor_types:
        n_sensors += 1

    assert len(cnt.ch_names) == n_sensors, (
        "Expected {:d} channel names, got {:d} channel names".format(
            n_sensors, len(cnt.ch_names)))

    # change from volt to micro-volt
    data = (cnt.get_data() * 1e6).astype(np.float32)
    fs = cnt.info['sfreq']
    log.info("Preprocessing...")
    for fn in preproc_functions:
        log.info(fn)
        data, fs = fn(data, fs)
        data = data.astype(np.float32)
        fs = float(fs)
    return data

class DiagnosisSet(object):
    def __init__(self, n_recordings, max_recording_mins, preproc_functions,
                 data_folders,
                 train_or_eval='train', sensor_types=['EEG'], ):
        self.n_recordings = n_recordings
        self.max_recording_mins = max_recording_mins
        self.preproc_functions = preproc_functions
        self.train_or_eval = train_or_eval
        self.sensor_types = sensor_types
        self.data_folders = data_folders

    def load(self, only_return_labels=False):
        log.info("Read file names")

        all_file_names, labels = get_all_sorted_file_names_and_labels(
            train_or_eval=self.train_or_eval,
            folders=self.data_folders, )

        if self.max_recording_mins is not None:
            log.info("Read recording lengths...")
            assert 'train' == self.train_or_eval
            # Computation as:
            lengths = [get_recording_length(fname) for fname in all_file_names]
            lengths = np.array(lengths)
            mask = lengths < self.max_recording_mins * 60
            cleaned_file_names = np.array(all_file_names)[mask]
            cleaned_labels = labels[mask]
        else:
            cleaned_file_names = np.array(all_file_names)
            cleaned_labels = labels
        if only_return_labels:
            return cleaned_labels
        X = []
        y = []
        n_files = len(cleaned_file_names[:self.n_recordings])
        for i_fname, fname in enumerate(cleaned_file_names[:self.n_recordings]):
            log.info("Load {:d} of {:d}".format(i_fname + 1, n_files))
            x = load_data(fname, preproc_functions=self.preproc_functions,
                          sensor_types=self.sensor_types)
            assert x is not None
            X.append(x)
            y.append(cleaned_labels[i_fname])
        y = np.array(y)
        return X, y

class CroppedDiagnosisMonitor(object):
    """
    Compute trialwise misclasses from predictions for crops for non-dense predictions.

    Parameters
    ----------
    input_time_length: int
        Temporal length of one input to the model.
    """

    def __init__(self, input_time_length, n_preds_per_input):
        self.input_time_length = input_time_length
        self.n_preds_per_input = n_preds_per_input

    def monitor_epoch(self, ):
        return

    def monitor_set(self, setname, all_preds, all_losses,
                    all_batch_sizes, all_targets, dataset):
        """Assuming one hot encoding for now"""
        preds_per_trial = compute_preds_per_trial(
            all_preds, dataset, input_time_length=self.input_time_length,
            n_stride=self.n_preds_per_input)

        mean_preds_per_trial = [np.mean(preds, axis=(0, 2)) for preds in
                                preds_per_trial]
        mean_preds_per_trial = np.array(mean_preds_per_trial)

        pred_labels_per_trial = np.argmax(mean_preds_per_trial, axis=1)
        assert pred_labels_per_trial.shape == dataset.y.shape
        accuracy = np.mean(pred_labels_per_trial == dataset.y)
        misclass = 1 - accuracy
        column_name = "{:s}_misclass".format(setname)
        out = {column_name: float(misclass)}
        y = dataset.y

        n_true_positive = np.sum((y == 1) & (pred_labels_per_trial == 1))
        n_positive = np.sum(y == 1)
        if n_positive > 0:
            sensitivity = n_true_positive / float(n_positive)
        else:
            sensitivity = np.nan
        column_name = "{:s}_sensitivity".format(setname)
        out.update({column_name: float(sensitivity)})

        n_true_negative = np.sum((y == 0) & (pred_labels_per_trial == 0))
        n_negative = np.sum(y == 0)
        if n_negative > 0:
            specificity = n_true_negative / float(n_negative)
        else:
            specificity = np.nan
        column_name = "{:s}_specificity".format(setname)
        out.update({column_name: float(specificity)})
        if (n_negative > 0) and (n_positive > 0):
            auc = roc_auc_score(y, mean_preds_per_trial[:, 1])
        else:
            auc = np.nan
        column_name = "{:s}_auc".format(setname)
        out.update({column_name: float(auc)})
        return out

def compute_preds_per_trial(preds_per_batch, dataset, input_time_length,
                            n_stride):
    n_trials = len(dataset.X)
    i_pred_starts = [input_time_length -
                     n_stride] * n_trials
    i_pred_stops = [t.shape[1] for t in dataset.X]

    start_stop_block_inds_per_trial = _compute_start_stop_block_inds(
        i_pred_starts,
        i_pred_stops, input_time_length, n_stride,
        False)

    n_rows_per_trial = [len(block_inds) for block_inds in
                        start_stop_block_inds_per_trial]

    all_preds_arr = np.concatenate(preds_per_batch, axis=0)
    i_row = 0
    preds_per_trial = []
    for n_rows in n_rows_per_trial:
        preds_per_trial.append(all_preds_arr[i_row:i_row + n_rows])
        i_row += n_rows
    assert i_row == len(all_preds_arr)
    return preds_per_trial

class CroppedNonDenseTrialMisclassMonitor(object):
    """
    Compute trial-wise misclasses from predictions for crops for non-dense predictions.

    Parameters
    ----------
    input_time_length: int
        Temporal length of one input to the model.
    """

    def __init__(self, input_time_length, n_preds_per_input):
        self.input_time_length = input_time_length
        self.n_preds_per_input = n_preds_per_input

    def monitor_set(self, setname, all_preds, all_losses,
                    all_batch_sizes, all_targets, dataset):
        """Assuming one hot encoding for now"""
        n_trials = len(dataset.X)
        i_pred_starts = [self.input_time_length -
                         self.n_preds_per_input] * n_trials
        i_pred_stops = [t.shape[1] for t in dataset.X]

        start_stop_block_inds_per_trial = _compute_start_stop_block_inds(
            i_pred_starts,
            i_pred_stops, self.input_time_length, self.n_preds_per_input,
            False)

        n_rows_per_trial = [len(block_inds) for block_inds in
                            start_stop_block_inds_per_trial]

        all_preds_arr = np.concatenate(all_preds, axis=0)
        i_row = 0
        preds_per_trial = []
        for n_rows in n_rows_per_trial:
            preds_per_trial.append(all_preds_arr[i_row:i_row + n_rows])
            i_row += n_rows

        mean_preds_per_trial = [np.mean(preds, axis=(0, 2)) for preds in
                                preds_per_trial]
        mean_preds_per_trial = np.array(mean_preds_per_trial)

        pred_labels_per_trial = np.argmax(mean_preds_per_trial, axis=1)
        assert pred_labels_per_trial.shape == dataset.y.shape
        accuracy = np.mean(pred_labels_per_trial == dataset.y)
        misclass = 1 - accuracy
        column_name = "{:s}_misclass".format(setname)
        return {column_name: float(misclass)}

class TrainValidTestSplitter(object):
    def __init__(self, n_folds, i_test_fold, shuffle):
        self.n_folds = n_folds
        self.i_test_fold = i_test_fold
        self.rng = RandomState(39483948)
        self.shuffle = shuffle

    def split(self, X, y, ):
        if len(X) < self.n_folds:
            raise ValueError("Less Trials: {:d} than folds: {:d}".format(
                len(X), self.n_folds
            ))
        folds = get_balanced_batches(len(X), self.rng, self.shuffle,
                                     n_batches=self.n_folds)
        test_inds = folds[self.i_test_fold]
        valid_inds = folds[self.i_test_fold - 1]
        all_inds = list(range(len(X)))
        train_inds = np.setdiff1d(all_inds, np.union1d(test_inds, valid_inds))
        assert np.intersect1d(train_inds, valid_inds).size == 0
        assert np.intersect1d(train_inds, test_inds).size == 0
        assert np.intersect1d(valid_inds, test_inds).size == 0
        assert np.array_equal(np.sort(
            np.union1d(train_inds, np.union1d(valid_inds, test_inds))),
            all_inds)

        train_set = create_set(X, y, train_inds)
        valid_set = create_set(X, y, valid_inds)
        test_set = create_set(X, y, test_inds)

        return train_set, valid_set, test_set

class TrainValidSplitter(object):
    def __init__(self, n_folds, i_valid_fold, shuffle):
        self.n_folds = n_folds
        self.i_valid_fold = i_valid_fold
        self.rng = RandomState(39483948)
        self.shuffle = shuffle

    def split(self, X, y):
        if len(X) < self.n_folds:
            raise ValueError("Less Trials: {:d} than folds: {:d}".format(
                len(X), self.n_folds
            ))
        folds = get_balanced_batches(len(X), self.rng, self.shuffle,
                                     n_batches=self.n_folds)
        valid_inds = folds[self.i_valid_fold]
        all_inds = list(range(len(X)))
        train_inds = np.setdiff1d(all_inds, valid_inds)
        assert np.intersect1d(train_inds, valid_inds).size == 0
        assert np.array_equal(np.sort(np.union1d(train_inds, valid_inds)),
                              all_inds)

        train_set = create_set(X, y, train_inds)
        valid_set = create_set(X, y, valid_inds)
        return train_set, valid_set

def run_exp(data_folders,
            n_recordings,
            sensor_types,
            n_chans,
            max_recording_mins,
            sec_to_cut, duration_recording_mins,
            test_recording_mins,
            max_abs_val,
            sampling_freq,
            divisor,
            test_on_eval,
            n_folds, i_test_fold,
            shuffle,
            model_name,
            n_start_chans, n_chan_factor,
            input_time_length, final_conv_length,
            model_constraint,
            init_lr,
            batch_size, max_epochs, cuda, ):
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    preproc_functions = []

    preproc_functions.append(
        lambda data, fs: (data[:, int(sec_to_cut * fs):-int(
            sec_to_cut * fs)], fs))
    preproc_functions.append(
        lambda data, fs: (data[:, :int(duration_recording_mins * 60 * fs)], fs))

    if max_abs_val is not None:
        preproc_functions.append(lambda data, fs:
                                 (np.clip(data, -max_abs_val, max_abs_val), fs))

    preproc_functions.append(lambda data, fs: (resampy.resample(data, fs,
                                                                sampling_freq,
                                                                axis=1,
                                                                filter='kaiser_fast'),
                                               sampling_freq))

    if divisor is not None:
        preproc_functions.append(lambda data, fs: (data / divisor, fs))

    dataset = DiagnosisSet(n_recordings=n_recordings,
                           max_recording_mins=max_recording_mins,
                           preproc_functions=preproc_functions,
                           data_folders=data_folders,
                           train_or_eval='train',
                           sensor_types=sensor_types)

    if test_on_eval:
        if test_recording_mins is None:
            test_recording_mins = duration_recording_mins
        test_preproc_functions = copy(preproc_functions)
        test_preproc_functions[1] = lambda data, fs: (
            data[:, :int(test_recording_mins * 60 * fs)], fs)
        test_dataset = DiagnosisSet(n_recordings=n_recordings,
                                    max_recording_mins=None,
                                    preproc_functions=test_preproc_functions,
                                    data_folders=data_folders,
                                    train_or_eval='eval',
                                    sensor_types=sensor_types)

    X, y = dataset.load()
    max_shape = np.max([list(x.shape) for x in X], axis=0)

    assert max_shape[1] == int(duration_recording_mins *
                               sampling_freq * 60)

    if test_on_eval:
        test_X, test_y = test_dataset.load()
        max_shape = np.max([list(x.shape) for x in test_X],
                           axis=0)
        assert max_shape[1] == int(test_recording_mins *
                                   sampling_freq * 60)
    if not test_on_eval:
        splitter = TrainValidTestSplitter(n_folds, i_test_fold,
                                          shuffle=shuffle)
        train_set, valid_set, test_set = splitter.split(X, y)
    else:
        splitter = TrainValidSplitter(n_folds, i_valid_fold=i_test_fold,
                                      shuffle=shuffle)
        train_set, valid_set = splitter.split(X, y)
        test_set = SignalAndTarget(test_X, test_y)
        del test_X, test_y
    del X, y  # shouldn't be necessary, but just to make sure

    set_random_seeds(seed=20170629, cuda=cuda)
    n_classes = 2
    if model_name == 'shallow':
        model = ShallowFBCSPNet(in_chans=n_chans, n_classes=n_classes,
                                n_filters_time=n_start_chans,
                                n_filters_spat=n_start_chans,
                                input_time_length=input_time_length,
                                final_conv_length=final_conv_length).create_network()
    elif model_name == 'deep':
        model = Deep4Net(n_chans, n_classes,
                         n_filters_time=n_start_chans,
                         n_filters_spat=n_start_chans,
                         input_time_length=input_time_length,
                         n_filters_2=int(n_start_chans * n_chan_factor),
                         n_filters_3=int(n_start_chans * (n_chan_factor ** 2.0)),
                         n_filters_4=int(n_start_chans * (n_chan_factor ** 3.0)),
                         final_conv_length=final_conv_length,
                         stride_before_pool=True).create_network()

    elif (model_name == 'deep_smac'):
        if model_name == 'deep_smac':
            do_batch_norm = False
        else:
            assert model_name == 'deep_smac_bnorm'
            do_batch_norm = True
        double_time_convs = False
        drop_prob = 0.244445
        filter_length_2 = 12
        filter_length_3 = 14
        filter_length_4 = 12
        filter_time_length = 21
        final_conv_length = 1
        first_nonlin = elu
        first_pool_mode = 'mean'
        first_pool_nonlin = identity
        later_nonlin = elu
        later_pool_mode = 'mean'
        later_pool_nonlin = identity
        n_filters_factor = 1.679066
        n_filters_start = 32
        pool_time_length = 1
        pool_time_stride = 2
        split_first_layer = True
        n_chan_factor = n_filters_factor
        n_start_chans = n_filters_start
        model = Deep4Net(n_chans, n_classes,
                         n_filters_time=n_start_chans,
                         n_filters_spat=n_start_chans,
                         input_time_length=input_time_length,
                         n_filters_2=int(n_start_chans * n_chan_factor),
                         n_filters_3=int(n_start_chans * (n_chan_factor ** 2.0)),
                         n_filters_4=int(n_start_chans * (n_chan_factor ** 3.0)),
                         final_conv_length=final_conv_length,
                         batch_norm=do_batch_norm,
                         double_time_convs=double_time_convs,
                         drop_prob=drop_prob,
                         filter_length_2=filter_length_2,
                         filter_length_3=filter_length_3,
                         filter_length_4=filter_length_4,
                         filter_time_length=filter_time_length,
                         first_nonlin=first_nonlin,
                         first_pool_mode=first_pool_mode,
                         first_pool_nonlin=first_pool_nonlin,
                         later_nonlin=later_nonlin,
                         later_pool_mode=later_pool_mode,
                         later_pool_nonlin=later_pool_nonlin,
                         pool_time_length=pool_time_length,
                         pool_time_stride=pool_time_stride,
                         split_first_layer=split_first_layer,
                         stride_before_pool=True).create_network()
    elif model_name == 'shallow_smac':
        conv_nonlin = identity
        do_batch_norm = True
        drop_prob = 0.328794
        filter_time_length = 56
        final_conv_length = 22
        n_filters_spat = 73
        n_filters_time = 24
        pool_mode = 'max'
        pool_nonlin = identity
        pool_time_length = 84
        pool_time_stride = 3
        split_first_layer = True
        model = ShallowFBCSPNet(in_chans=n_chans, n_classes=n_classes,
                                n_filters_time=n_filters_time,
                                n_filters_spat=n_filters_spat,
                                input_time_length=input_time_length,
                                final_conv_length=final_conv_length,
                                conv_nonlin=conv_nonlin,
                                batch_norm=do_batch_norm,
                                drop_prob=drop_prob,
                                filter_time_length=filter_time_length,
                                pool_mode=pool_mode,
                                pool_nonlin=pool_nonlin,
                                pool_time_length=pool_time_length,
                                pool_time_stride=pool_time_stride,
                                split_first_layer=split_first_layer,
                                ).create_network()
    elif model_name == 'linear':
        model = nn.Sequential()
        model.add_module("conv_classifier",
                         nn.Conv2d(n_chans, n_classes, (600, 1)))
        model.add_module('softmax', nn.LogSoftmax())
        model.add_module('squeeze', Expression(lambda x: x.squeeze(3)))
    else:
        assert False, "unknown model name {:s}".format(model_name)
    to_dense_prediction_model(model)
    log.info("Model:\n{:s}".format(str(model)))
    if cuda:
        model.cuda()
    # determine output size
    test_input = np_to_var(
        np.ones((2, n_chans, input_time_length, 1), dtype=np.float32))
    if cuda:
        test_input = test_input.cuda()
    log.info("In shape: {:s}".format(str(test_input.cpu().data.numpy().shape)))

    out = model(test_input)
    log.info("Out shape: {:s}".format(str(out.cpu().data.numpy().shape)))
    n_preds_per_input = out.cpu().data.numpy().shape[2]
    log.info("{:d} predictions per input/trial".format(n_preds_per_input))
    iterator = CropsFromTrialsIterator(batch_size=batch_size,
                                       input_time_length=input_time_length,
                                       n_preds_per_input=n_preds_per_input)
    optimizer = optim.Adam(model.parameters(), lr=init_lr)

    loss_function = lambda preds, targets: F.nll_loss(
        th.mean(preds, dim=2, keepdim=False), targets)

    if model_constraint is not None:
        assert model_constraint == 'defaultnorm'
        model_constraint = MaxNormDefaultConstraint()
    monitors = [LossMonitor(), MisclassMonitor(col_suffix='sample_misclass'),
                CroppedDiagnosisMonitor(input_time_length, n_preds_per_input),
                RuntimeMonitor(), ]
    stop_criterion = MaxEpochs(max_epochs)
    batch_modifier = None
    run_after_early_stop = True
    exp = Experiment(model, train_set, valid_set, test_set, iterator,
                     loss_function, optimizer, model_constraint,
                     monitors, stop_criterion,
                     remember_best_column='valid_misclass',
                     run_after_early_stop=run_after_early_stop,
                     batch_modifier=batch_modifier,
                     cuda=cuda)
    exp.run()
    return exp

if __name__ == "__main__":
    import config

    start_time = time.time()
    logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                        level=logging.DEBUG, stream=sys.stdout)
    exp = run_exp(
        config.data_folders,
        config.n_recordings,
        config.sensor_types,
        config.n_chans,
        config.max_recording_mins,
        config.sec_to_cut, config.duration_recording_mins,
        config.test_recording_mins,
        config.max_abs_val,
        config.sampling_freq,
        config.divisor,
        config.test_on_eval,
        config.n_folds, config.i_test_fold,
        config.shuffle,
        config.model_name,
        config.n_start_chans, config.n_chan_factor,
        config.input_time_length, config.final_conv_length,
        config.model_constraint,
        config.init_lr,
        config.batch_size, config.max_epochs, config.cuda, )
    end_time = time.time()
    run_time = end_time - start_time

    log.info("Experiment runtime: {:.2f} sec".format(run_time))

    # make trial-wise predictions per recording, compute the mean prediction probability, get final prediction
    exp.model.eval()
    for setname in ('train', 'valid', 'test'):
        log.info("Compute predictions for {:s}...".format(
            setname))
        dataset = exp.datasets[setname]
        if config.cuda:
            preds_per_batch = [var_to_np(exp.model(np_to_var(b[0]).cuda()))
                               for b in exp.iterator.get_batches(dataset, shuffle=False)]
        else:
            preds_per_batch = [var_to_np(exp.model(np_to_var(b[0])))
                               for b in exp.iterator.get_batches(dataset, shuffle=False)]

        preds_per_trial = compute_preds_per_trial(
            preds_per_batch, dataset,
            input_time_length=exp.iterator.input_time_length,
            n_stride=exp.iterator.n_preds_per_input)

        mean_preds_per_trial = [np.mean(preds, axis=(0, 2)) for preds in
                                preds_per_trial]

        mean_preds_per_trial = np.array(mean_preds_per_trial)
