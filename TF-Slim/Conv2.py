import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.contrib.slim as slim

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

BATCH_SIZE = 100
TRAINING_STEPS = 30000


def inference(input_tensor):
    input_tensor = tf.reshape(input_tensor, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                        biases_initializer=tf.constant_initializer(0.1),
                        weights_regularizer=slim.l2_regularizer(0.0005)):

        op = slim.conv2d(input_tensor, CONV1_DEEP, [CONV1_SIZE, CONV1_SIZE], scope='layer1-conv')
        op = slim.max_pool2d(op, [2, 2], 2, padding='SAME', scope='layer2-max-pool')
        # op = slim.conv2d(op, CONV2_DEEP, [CONV2_SIZE, CONV2_SIZE], scope='layer3-conv')
        # op = slim.max_pool2d(op, [2, 2], 2, padding='SAME', scope='layer4-max-pool')
        op = slim.flatten(op, scope='flatten')
        op = slim.fully_connected(op, FC_SIZE, scope="layer5-fc")
        output = slim.fully_connected(op, OUTPUT_NODE, scope="output")
    return output


def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

    y = inference(x)
    slim.losses.softmax_cross_entropy(y, y_)
    total_loss = slim.losses.get_total_loss()
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_tensor = slim.learning.create_train_op(total_loss, optimizer)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        test_feed ={x: mnist.test.images, y_: mnist.test.labels}

        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print(i, validate_acc)

            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_tensor, feed_dict={x: xs, y_: ys})

        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print(TRAINING_STEPS, test_acc)


def main():
    mnist = input_data.read_data_sets('/path/to/MNIST_data', one_hot=True)
    train(mnist)


if __name__ == '__main__':
    main()

