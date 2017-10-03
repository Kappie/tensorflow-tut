    def my_model_fn(features, labels, mode, params):
        M, N = features.get_shape().as_list()[-2:]
        scattering_coefficients = Scattering(M=M, N=N, J=1, L=2)(features)
        batch_size = scattering_coefficients.get_shape().as_list()[0]
        # throw all coefficients into single vector for each image
        scattering_coefficients = tf.reshape(scattering_coefficients, [batch_size, -1])
        n_classes = 10
        n_coefficients = scattering_coefficients.get_shape().as_list()[1]
        # use linear classifier
        W = tf.Variable(tf.zeros([n_coefficients, n_classes]))
        b = tf.Variable(tf.zeros([n_classes]))
        y_predict = tf.nn.softmax(tf.matmul(scattering_coefficients, W) + b)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions={"predictions": y_predict})

        # loss function and training step
        cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=y_predict) )
        train_op = tf.train.GradientDescentOptimizer(params["learning_rate"]).minimize(cross_entropy)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=cross_entropy,
            train_op=train_op)


    def sample_batch(X, y, batch_size):
        idx = np.random.choice(X.shape[0], batch_size, replace=False)
        return tf.convert_to_tensor(X[idx]), tf.convert_to_tensor(y[idx])

    LEARNING_RATE = 0.01
    BATCH_SIZE = 2
    n_training_steps = 2
    image_dimension = 28
    model_params = {"learning_rate": LEARNING_RATE}

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    X_train = mnist.train.images.astype(np.float32)
    X_train = normalize(X_train)
    # number of channels is 1, -1 infers number of samples
    X_train = X_train.reshape(-1, 1, image_dimension, image_dimension)
    y_train = mnist.train.labels.astype(np.int64)

    X_validation = mnist.validation.images.astype(np.float32)
    X_validation = normalize(X_validation)
    X_validation = X_validation.reshape(-1, 1, image_dimension, image_dimension)
    y_validation = mnist.validation.labels.astype(np.int64)

    train_input_fn = lambda: sample_batch(X_train, y_train, BATCH_SIZE)
    validation_input_fn = lambda: sample_batch(X_validation, y_validation, BATCH_SIZE)

    # Train
    scattering_classifier = tf.estimator.Estimator(model_fn=my_model_fn, params=model_params)
    # Hangs forever...
    scattering_classifier.train(input_fn=train_input_fn, max_steps=n_training_steps)
