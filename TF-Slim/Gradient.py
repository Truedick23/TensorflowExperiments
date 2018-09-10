import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.contrib.slim as slim

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500
BATCH_SIZE = 100
LEARNING_RATE = 0.03
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 20000
MOVING_AVERAGE_DECAY = 0.99


def inference(inputs):

    layer1 = slim.fully_connected(inputs=inputs,
                                  num_outputs=LAYER1_NODE,
                                  weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                  biases_initializer=tf.constant_initializer(0.1),
                                  activation_fn=tf.nn.relu)

    outputs = slim.fully_connected(inputs=layer1,
                                   num_outputs=OUTPUT_NODE,
                                   weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                   biases_initializer=tf.constant_initializer(0.1),
                                   activation_fn=tf.nn.relu)

    return outputs

def train(mnist):
    x = tf.placeholder(tf.float32, shape=[None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, shape=[None, OUTPUT_NODE], name='y-input')
    y = inference(x)

    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    '''
    regularizer = tf.contrib.layers.l2_regulatizer(REGULARIZATION_RATE)
    regularization = regularizer(weights1) + regularizer(weights2)
    '''

    loss = cross_entropy_mean
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss, global_step=global_step)

    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print(i, validate_acc)

            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print(TRAINING_STEPS, test_acc)


def main(argv=None):
    mnist = input_data.read_data_sets("/path/to/MNIST_data", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    main()


