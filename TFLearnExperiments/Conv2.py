import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import tflearn.datasets.mnist as mnist

INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

CONV1_DEEP = 32
CONV1_SIZE = 5
CONV2_DEEP = 64
CONV2_SIZE = 5
FC_SIZE = 512

trainX, trainY, testX, testY = mnist.load_data(data_dir='/path/to/MNIST_data', one_hot=True)

trainX = trainX.reshape([-1, IMAGE_SIZE, IMAGE_SIZE, 1])
testX = testX.reshape([-1, IMAGE_SIZE, IMAGE_SIZE, 1])

net = input_data(shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name='input')
net = conv_2d(net, CONV1_DEEP, CONV1_SIZE, weights_init='truncated_normal', bias_init='zeros', activation='relu')
tflearn.helpers.regularizer.add_weights_regularizer(net, loss='L2', weight_decay=0.001)
net = max_pool_2d(net, 2)
net = conv_2d(net, CONV2_DEEP, CONV2_SIZE, weights_init='truncated_normal', bias_init='zeros', activation='relu')
tflearn.helpers.regularizer.add_weights_regularizer(net, loss='L2', weight_decay=0.001)
net = max_pool_2d(net, 2)
net = fully_connected(net, FC_SIZE, activation='relu', bias_init=tf.constant_initializer(0.1))
tflearn.helpers.regularizer.add_weights_regularizer(net, loss='L2', weight_decay=0.001)
output = fully_connected(net, NUM_LABELS, bias_init=tf.constant_initializer(0.1))
tflearn.helpers.regularizer.add_weights_regularizer(output, loss='L2', weight_decay=0.001)

output = regression(output, optimizer='Adam', learning_rate=0.01, loss='categorical_crossentropy')
model = tflearn.DNN(output, tensorboard_verbose=0)
model.fit(trainX, trainY, n_epoch=20, validation_set=([testX, testY]), show_metric=True)

score = model.evaluate(testX, testY)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])