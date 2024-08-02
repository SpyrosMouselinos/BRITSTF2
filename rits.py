import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Dense, Dropout, Concatenate


class ZeroDiagonalConstraint(tf.keras.constraints.Constraint):
    """
    Custom Implementation of the Zero diagonal Constraint
    """

    def __init__(self):
        return

    def call(self, w):
        """Return the modified weight matrix by subtracting its diagonal.

        This function takes a weight matrix as input and computes a new matrix
        by subtracting the diagonal elements from the original matrix. The
        resulting matrix can be used as a constraint matrix for various
        applications in linear algebra or machine learning.

        Args:
            w (tf.Tensor): The weight matrix from which the diagonal will be subtracted.

        Returns:
            tf.Tensor: The constraint matrix obtained after subtracting the diagonal from the
                weight matrix.
        """
        w = w - tf.linalg.diag(w)
        return w


class FeatureRegression(Layer):
    def __init__(self, name="FeatureRegression"):
        super(FeatureRegression, self).__init__(name=name)

    def build(self, input_shape):
        """Build the weights and biases for a layer.

        This method initializes the weights and biases for the layer based on
        the provided input shape. The weights are created with a shape that
        matches the last dimension of the input shape, and a zero diagonal
        constraint is applied to ensure that the diagonal elements are zero. The
        biases are initialized to zeros.

        Args:
            input_shape (tuple): The shape of the input data, where the last dimension is used to define
                the shape of the weights and biases.

        Returns:
            None: This method does not return any value, but it initializes the layer's
                weights and biases.
        """

        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1]), dtype='float32', name='FR_W',
                                 constraint=ZeroDiagonalConstraint(), initializer='he_uniform')
        self.b = self.add_weight(shape=input_shape[-1], dtype='float32', name='FR_b', initializer='zeros')

    @tf.function
    def call(self, inputs, *args, **kwargs):
        """Perform a matrix multiplication and add a bias term.

        This function takes input data and computes the result of a matrix
        multiplication with weights, followed by the addition of a bias vector.
        It utilizes TensorFlow's matmul function to perform the multiplication.
        The resulting tensor can be used in various neural network operations.

        Args:
            inputs (Tensor): The input tensor to be multiplied.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Tensor: The result of the matrix multiplication and bias addition.
        """

        z_h = tf.matmul(inputs, self.W, transpose_b=True) + self.b
        return z_h


class TemporalDecay(Layer):
    def __init__(self, units, diag=False, name='TemporalDecay'):
        super(TemporalDecay, self).__init__(name=name)
        self.units = units
        self.diag = diag
        return

    def build(self, input_shape):
        """Build the weight and bias tensors for a layer.

        This method initializes the weight matrix and bias vector based on the
        provided input shape. If the `diag` attribute is set to True, it ensures
        that the weight matrix has a zero diagonal constraint. The weight matrix
        is initialized using the 'he_uniform' initializer, which is suitable for
        layers with ReLU activation functions. The bias vector is initialized to
        zeros.

        Args:
            input_shape (tuple): The shape of the input tensor, where the last
                dimension should match the number of units.
        """

        if self.diag:
            assert (self.units == input_shape[-1])
            self.W = self.add_weight(shape=(self.units, input_shape[-1]), dtype='float32', name='TD_W',
                                     initializer='he_uniform', constraint=ZeroDiagonalConstraint())
        else:
            self.W = self.add_weight(shape=(self.units, input_shape[-1]), dtype='float32', name='TD_W',
                                     initializer='he_uniform')

        self.b = self.add_weight(shape=self.units, dtype='float32', name='TD_b', initializer='zeros')

    @tf.function
    def call(self, inputs, *args, **kwargs):
        """Compute the output of a neural network layer using a linear
        transformation followed by a non-linear activation.

        This function performs a matrix multiplication of the input with
        weights, adds a bias, applies the ReLU activation function, and then
        computes the exponential of the negative result. This is typically used
        in the context of neural networks to transform inputs into outputs.

        Args:
            inputs (tf.Tensor): The input tensor to the layer.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            tf.Tensor: The transformed output tensor after applying the operations.
        """

        gamma = tf.nn.relu(tf.matmul(inputs, self.W, transpose_b=True) + self.b)
        gamma = tf.math.exp(-gamma)
        return gamma


class RITS(Model):
    def __init__(self, internal_dim, hid_dim, sequence_length=None, go_backwards=False, name="Rits"):
        super(RITS, self).__init__(name=name)
        self.hid_dim = hid_dim
        self.internal_dim = internal_dim
        self.sequence_length = sequence_length
        self.go_backwards = go_backwards
        return

    def build(self, input_shape):
        """Build the layers of the model.

        This method initializes various layers of a neural network model,
        including LSTM cells, temporal decay layers, dense layers for
        regression, and an output layer. It sets up the architecture based on
        the specified input shape, which determines the sequence length for
        processing time-series data.

        Args:
            input_shape (tuple): A tuple representing the shape of the input data,
                where the second element indicates the sequence length.
        """

        self.rnn_cell = tf.keras.layers.LSTM(units=self.hid_dim, return_state=True)
        self.temp_decay_h = TemporalDecay(units=self.hid_dim, diag=False)
        self.temp_decay_x = TemporalDecay(units=self.internal_dim, diag=True)
        self.hist_reg = Dense(units=self.internal_dim, activation='linear')
        self.feat_reg = FeatureRegression()
        self.weight_combine = Dense(units=self.internal_dim, activation='linear')
        self.dense = Dense(units=self.internal_dim, activation='relu')
        self.out = Dense(units=1, activation='linear')
        self.sequence_length = input_shape[1]

    @tf.function
    def call(self, values, masks, deltas):
        """Call the RNN cell with input values, masks, and deltas to produce
        predictions and imputations.

        This method processes a sequence of input values through an RNN cell,
        applying various transformations and loss calculations at each time
        step. It maintains hidden and cell states, applies decay functions based
        on the deltas, and computes custom losses based on the differences
        between the input values and their historical or corrected versions. The
        final output includes predictions from the RNN, imputations for the
        input values, and a custom loss metric.

        Args:
            values (tf.Tensor): A tensor of shape (batch_size, sequence_length, features) representing
                the input values.
            masks (tf.Tensor): A tensor of shape (batch_size, sequence_length, features) used to mask
                the input values.
            deltas (tf.Tensor): A tensor of shape (batch_size, sequence_length, features) representing
                the deltas for decay calculations.

        Returns:
            tuple: A tuple containing:
                - predictions (tf.Tensor): The output predictions from the RNN.
                - imputations (tf.Tensor): The imputations for the input values.
                - custom_loss (tf.Tensor): The computed custom loss over the sequence.
        """

        h = tf.zeros(shape=(values.shape[0], self.hid_dim))
        c = tf.zeros(shape=(values.shape[0], self.hid_dim))
        imputations = []
        custom_loss = []
        for t in range(self.sequence_length):

            if self.go_backwards:
                x = values[:, self.sequence_length - t - 1, :]
                m = masks[:, self.sequence_length - t - 1, :]
                d = deltas[:, self.sequence_length - t - 1, :]
            else:
                x = values[:, t, :]
                m = masks[:, t, :]
                d = deltas[:, t, :]

            ### History and Input Decay Conditioned on Deltas
            gamma_h = self.temp_decay_h(d)  # Page 4 equation(3)
            h = h * gamma_h
            x_hat = self.hist_reg(h)  # Page 4 Equation (1)

            ### Loss 1: Absolute Error Between Input X(t) and Historical Decayed Input X_H(t-1)
            custom_loss_x = tf.reduce_sum((tf.abs(x - x_hat) * m) / (tf.reduce_sum(m) + 1e-6), axis=1)
            x_c = m * x + (1 - m) * x_hat  # Page 4 Equation (2)

            z_hat = self.feat_reg(x_c)  # Page 5 Equation (7)

            ### Loss 2: Relative Error Between Input X(t) and Zeta Hat
            custom_loss_x += tf.reduce_sum((tf.abs(x - z_hat) * m) / (tf.reduce_sum(m) + 1e-6), axis=1)

            gamma_x = self.temp_decay_x(d)  # Page 4 equation(3)
            beta = self.weight_combine(tf.concat([gamma_x, m], axis=1))  # Page 6 Equation (8)
            c_hat = beta * z_hat + (1 - beta) * x_hat

            ### Loss 3: Relative Error Between Input X(t) and the feature correlated corrected Input
            custom_loss_x += tf.reduce_sum((tf.abs(x - c_hat) * m) / (tf.reduce_sum(m) + 1e-6), axis=1)
            c_c = m * x + (1 - m) * c_hat

            imputations.append(c_c)
            inputs = tf.concat([c_c, m], axis=1)

            _, h, c = self.rnn_cell(tf.expand_dims(inputs, axis=1), [h, c])
            custom_loss.append(custom_loss_x)
        imputations = tf.concat([tf.expand_dims(f, axis=1) for f in imputations], axis=1)
        predictions = self.dense(h)
        predictions = self.out(predictions)
        custom_loss = tf.concat([tf.expand_dims(f, axis=1) for f in custom_loss], axis=1)
        custom_loss = tf.reduce_mean(custom_loss, axis=1)
        return predictions, imputations, custom_loss

# print("Debug Ready")
# x = tf.ones(shape=(7, 3, 5))
# m = tf.zeros(shape=(7, 3, 5))
# d = tf.zeros(shape=(7, 3, 5))
# opt = tf.keras.optimizers.Adam()
# rit_model = RITS(5, 100)
#
#
# @tf.function
# def train_step(x, m, d):
#     with tf.GradientTape() as tape:
#         predictions, imputations, custom_loss = rit_model(x, m, d)
#         loss = tf.keras.losses.mean_squared_error(0, 0.3*predictions + custom_loss)
#     gradients = tape.gradient(loss, rit_model.trainable_variables)
#     opt.apply_gradients(zip(gradients, rit_model.trainable_variables))
#     tf.print(loss)