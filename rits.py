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
        """
        Return the 0 diag weight matrix
        :param w: The weight matrix
        :return: The constraint matrix
        """
        w = w - tf.linalg.diag(w)
        return w


class FeatureRegression(Layer):
    def __init__(self, name="FeatureRegression"):
        super(FeatureRegression, self).__init__(name=name)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1]), dtype='float32', name='FR_W',
                                 constraint=ZeroDiagonalConstraint(), initializer='he_uniform')
        self.b = self.add_weight(shape=input_shape[-1], dtype='float32', name='FR_b', initializer='zeros')

    @tf.function
    def call(self, inputs, *args, **kwargs):
        z_h = tf.matmul(inputs, self.W, transpose_b=True) + self.b
        return z_h


class TemporalDecay(Layer):
    def __init__(self, units, diag=False, name='TemporalDecay'):
        super(TemporalDecay, self).__init__(name=name)
        self.units = units
        self.diag = diag
        return

    def build(self, input_shape):
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
            # custom_loss.append(custom_loss_x)
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


# @tf.function
# def train_step(x, m, d):
#     with tf.GradientTape() as tape:
#         predictions, imputations, custom_loss = rit_model(x, m, d)
#         loss = tf.keras.losses.mean_squared_error(0, 0.3*predictions + custom_loss)
#     gradients = tape.gradient(loss, rit_model.trainable_variables)
#     opt.apply_gradients(zip(gradients, rit_model.trainable_variables))
#     tf.print(loss)
