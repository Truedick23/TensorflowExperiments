import tflearn
from tflearn.layers.recurrent import lstm
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression
import tflearn.datasets.mnist as mnist

trainX, trainY, testX, testY = mnist.load_data(data_dir='/path/to/MNIST_data', one_hot=True)
trainX = trainX.reshape([-1, 784, 1])
testX = testX.reshape([-1, 784, 1])

net = input_data(shape=[None, 784, 1], name='input')
net = lstm(net, 128, activation='relu')
output = fully_connected(net, 10)

output = regression(output, optimizer='Adam', learning_rate=0.03, loss='categorical_crossentropy')
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(trainX, trainY, n_epoch=20, validation_set=([testX, trainY]), batch_size=128, show_metric=True)

score = model.evaluate(testX, testY)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])