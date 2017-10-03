from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf


mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# hyper parameters
learning_rate = 0.5
num_steps = 1000
batch_size = 100
display_step = 100

# classifier parameters
num_classes = 10
num_input = 784


def model_fn(features, labels, mode, params):
    x = features["images"]

    W = tf.Variable(tf.zeros([num_input, num_classes]))
    b = tf.Variable(tf.zeros([num_classes]))
    y = tf.matmul(x, W) + b
    predictions = tf.argmax(y, axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cross_entropy, global_step=tf.train.get_global_step())
    accuracy_op = tf.metrics.accuracy(labels=tf.argmax(labels, axis=1), predictions=predictions)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=cross_entropy,
        train_op=train_op,
        eval_metric_ops={'accuracy': accuracy_op}
    )


model = tf.estimator.Estimator(model_fn, params={})
input_fn_train = tf.estimator.inputs.numpy_input_fn(
    x={"images": mnist.train.images}, y=mnist.train.labels,
    batch_size=batch_size, num_epochs=None, shuffle=True
)
input_fn_test = tf.estimator.inputs.numpy_input_fn(
    x={"images": mnist.test.images}, y=mnist.test.labels,
    batch_size=batch_size, shuffle=False
)

model.train(input_fn_train, steps=num_steps)
evaluation = model.evaluate(input_fn_test)

print("Testing accuracy: ", evaluation["accuracy"])
