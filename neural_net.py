from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np
import cifar as cf
import time

def _data():
    print("Loading data...")
    # loads the training data and the testing data. creates two arrays
    # for each set of data: 1) the images, an Nx32x32x3 array (which is normalized)
    # and 2) the labels, an array of size N with each value being a label (0-9)
    _data.train_data = cf.load_training_data("cifar-10-batches-py")
    _data.train_images = cf.normalize(_data.train_data["images"])
    _data.train_labels = np.array(_data.train_data["labels"])

    _data.test_data = cf.load_test_data("cifar-10-batches-py")
    _data.test_images = cf.normalize(_data.test_data["images"])
    _data.test_labels = np.array(_data.test_data["labels"])

def _feat_extract(images, num_bins):
    print("Extracting features...")
    # makes the feature vectors for the data sets. vectors are
    # initially Nx3, where N is the number of images in the set and
    # 3 are the mean color values for each image
    vector = cf.avg_color(images)

    if(num_bins == 0): return vector

    # appends the histogram data to each entry of the feature
    # vectors. this makes each vector of shape Nx67, as the histograms
    # are of length 64
    feats = np.empty(shape=(images.shape[0], num_bins ** 3))
    for i, img in enumerate(images):
        feats[i, :] = cf.color_hist(img, num_bins)
    vector = np.concatenate((vector, feats), axis = 1)

    return vector

def run(num_bins=-1):
    start = time.time()

    # extract features (or not) depending on number of bins
    if num_bins == -2:
        train_vector = _data.train_images
        test_vector = _data.test_images
    else:
        train_vector = _feat_extract(_data.train_images, num_bins)
        test_vector = _feat_extract(_data.test_images, num_bins)

    # create model. the model has an input layer (128), a hidden layer (256),
    # and an output layer (10)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(train_vector.shape[1:])),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['sparse_categorical_accuracy'])

    # train
    model.fit(train_vector, _data.train_labels, epochs=10, batch_size=500)

    # evaluate
    r = model.evaluate(test_vector, _data.test_labels, verbose = 2)

    return r[1], (time.time() - start)

_data()
xaxis = [-2, 0, 2, 4, 8, 16]
yaxis = []
fig = plt.figure()
ax = fig.add_subplot(111)

for i, num in enumerate(xaxis):
    y, t = run(num)
    yaxis.append(y)
    ax.annotate("%.2fs" % t, (num, y))

plt.plot(xaxis, yaxis)
plt.xlabel("Number of Bins")
plt.ylabel("Percent images classified correctly (%)")
plt.xticks([-1, 0, 2, 4, 8, 16])
plt.show()