from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
import keras
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def get_img_array(img_path, size):
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    array = keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array


def gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names):
    # create a CNN model that maps the input image to the activations
    # of the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

    # create a model that maps the activations of the last conv
    # layer to the final class predictions
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input

    # for layer_name in ['block5_pool', 'flatten', 'fc1', 'fc2', 'dense_1']:
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)

    # compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        # compute class predictions
        pred = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(pred[0])
        top_class_channel = pred[:, top_pred_index]

    # Get the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # reshape to a vector, where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # multiply each channel in the feature map array
    # with the feature importance with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # compute the channel-wise mean of the resulting feature map
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # normalize the heatmap between 0 and 1 for visualization purposes
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap


def show_feature_maps(img_path, img_size, model, last_conv_layer_name, classifier_layer_names):

    # prepare spectrogram image
    img_array = preprocess_input(get_img_array(img_path, size=img_size))

    # predict the class
    pred = model.predict(img_array)
    print('Class prediction [abnormal, normal] : ', pred)

    # generate class activation heatmap
    heatmap = gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names)

    # display heatmap
    #plt.matshow(heatmap)
    #plt.show()

    # load the original image
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)

    # rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # add heatmap to original image
    superimposed_img = jet_heatmap * 0.9 + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # return GradCam visualization
    return superimposed_img


