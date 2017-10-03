import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
NUMBER_OF_CLASSES = 10
NUMBER_OF_PIXELS = 784

x = tf.placeholder(tf.float32, [None, NUMBER_OF_PIXELS])
W = tf.Variable(tf.zeros([NUMBER_OF_PIXELS, NUMBER_OF_CLASSES]))
b = tf.Variable(tf.zeros([NUMBER_OF_CLASSES]))
y = tf.matmul(x, W) + b

y_true = tf.placeholder(tf.float32, [None, NUMBER_OF_CLASSES])
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(y), reduction_indices=[1]))
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y))

learning_rate = 0.5
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_true: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_true: mnist.test.labels}))
