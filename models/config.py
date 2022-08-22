# config file for the end-to-end deep4 model
data_folders = [
    '.../normal/',
    '.../abnormal/']
n_recordings = 10
sensor_types = ["EEG"]
n_chans = 19
max_recording_mins = 30
sec_to_cut = 60
duration_recording_mins = 20
test_recording_mins = 20
max_abs_val = 800
sampling_freq = 250
divisor = 10
test_on_eval = True
n_folds = 8
i_test_fold = 4
shuffle = True
model_name = 'deep'
n_start_chans = 25
n_chan_factor = 2
input_time_length = 6000
final_conv_length = 1
model_constraint = 'defaultnorm'
init_lr = 1e-3
batch_size = 64
max_epochs = 35
cuda = True
