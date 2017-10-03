import os
import urllib.request
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from scattering import Scattering
from tensorflow.examples.tutorials.mnist import input_data


def normalize(data_split):
    # Normalize pixel values between -1 and 1.
    return (data_split - 0.5) / 0.5

def my_model_fn(features, labels, mode, params):
    """
    Do the following:
    1. configure model via tensorflow operations
    2. define loss function for training/evaluation
    3. define training operation/optimizer
    4. generate predictions
    5. return predictions/loss/train_op/eval_metric_ops in EstimatorSpec object

    features: dict containing features passed via input_fn
    labels: Tensor containing labels passed to the model via input_fn
    mode: train, evaluate or predict
    params: additional hyperparameters
    """

    print(features)
    features = features["x"]
    print(features)
    M, N = features.get_shape().as_list()[-2:]
    print("let's scatter!")
    scattering_coefficients = Scattering(M=M, N=N, J=2, L=8)(features)
    print(scattering_coefficients)
    # batch_size = scattering_coefficients.get_shape().as_list()[0]
    # throw all coefficients into single vector for each image
    # scattering_coefficients = tf.reshape(scattering_coefficients, [batch_size, -1])
    dimensions = scattering_coefficients.get_shape().as_list()[1:]
    scattering_coefficients = tf.reshape(scattering_coefficients, [-1, np.prod(dimensions)])
    print(scattering_coefficients)
    n_classes = 10
    n_coefficients = scattering_coefficients.get_shape().as_list()[1]

    # use linear classifier
    W = tf.Variable(tf.zeros([n_coefficients, n_classes]))
    b = tf.Variable(tf.zeros([n_classes]))
    y_predict = tf.nn.softmax(tf.matmul(scattering_coefficients, W) + b)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions={"prediction": y_predict})

    # loss function and training step
    cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=y_predict) )
    train_op = tf.train.GradientDescentOptimizer(params["learning_rate"]).minimize(
        cross_entropy, global_step=tf.train.get_global_step())

    # train_op = tf.train.GradientDescentOptimizer(learning_rate=params["learning_rate"]).minimize(
    #     loss=loss)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=cross_entropy,
        train_op=train_op)


def calculate_accuracy(y_true, y_predicted):
    n_samples = 0
    n_correct = 0

    for index, prediction in enumerate(y_predicted):
        prediction = prediction["prediction"]
        predicted_class = np.argmax(prediction)
        true_class = np.argmax(y_true[index])
        if predicted_class == true_class:
            n_correct += 1
        n_samples += 1

    return (n_correct/n_samples, n_samples)



# def sample_batch(X, y, batch_size):
#     """
#     Returns Tensors feature_cols, labels
#     """
#     print("inputting sample batch")
#     idx = np.random.choice(X.shape[0], batch_size, replace=False)
#     return {"x": tf.convert_to_tensor(X[idx])}, tf.convert_to_tensor(y[idx])


LEARNING_RATE = 0.1
BATCH_SIZE = 128
n_training_steps = 1000
image_dimension = 28
n_classes = 10
model_params = {"learning_rate": LEARNING_RATE}

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# X_tensor = tf.placeholder(tf.float32, shape=[batch_size, 1, image_dimension, image_dimension])
# y_tensor = tf.placeholder(tf.int64, shape=[batch_size, n_classes])

X_train = mnist.train.images.astype(np.float32)
X_train = normalize(X_train)
# number of channels is 1, -1 infers number of samples
# X_train = tf.reshape(X_train, (-1, 1, image_dimension, image_dimension))
X_train = X_train.reshape(-1, 1, image_dimension, image_dimension)
y_train = mnist.train.labels.astype(np.int64)

X_validation = mnist.validation.images.astype(np.float32)
X_validation = normalize(X_validation)
# X_validation = tf.reshape(X_validation, (-1, 1, image_dimension, image_dimension))
X_validation = X_validation.reshape(-1, 1, image_dimension, image_dimension)
y_validation = mnist.validation.labels.astype(np.int64)

print(y_validation)

# train_input_fn = lambda: sample_batch(X_train, y_train, BATCH_SIZE)
# validation_input_fn = lambda: sample_batch(X_validation, y_validation, BATCH_SIZE)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_train},
    y=y_train,
    num_epochs=None,
    shuffle=True,
    batch_size=BATCH_SIZE
)
validation_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_validation},
    batch_size=X_validation.shape[0],
    num_epochs=1,
    shuffle=False
)

# Train
scattering_classifier = tf.estimator.Estimator(model_fn=my_model_fn, params=model_params)
print("start training")
scattering_classifier.train(input_fn=train_input_fn, steps=n_training_steps)

# Score accuracy
print("start scoring accuracy")
predictions = scattering_classifier.predict(input_fn=validation_input_fn)

print(calculate_accuracy(y_validation, predictions))
