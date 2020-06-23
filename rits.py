import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Dense, Dropout
from tensorflow.keras.layers import LSTMCell
import math
import argparse
from sklearn import metrics

SEQ_LEN = 49
RNN_HID_SIZE = 64


class ZeroDiagonalConstraint(tf.keras.constraints.Constraint):
    """
    Custom Implementation of the Zero diagonal Constraint
    """

    def __init__(self):
        return

    def __call__(self, w):
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

    # @tf.function
    def call(self, inputs, *args, **kwargs):
        gamma = tf.nn.relu(tf.matmul(inputs, self.W, transpose_b=True) + self.b)
        gamma = tf.math.exp(-gamma)
        return gamma


class RITS(Layer):
    def __init__(self, internal_dim, hid_dim, sequence_length=None, name="Rits"):
        super(RITS, self).__init__(name=name)
        self.hid_dim = hid_dim
        self.internal_dim = internal_dim
        self.sequence_length = sequence_length
        return

    def build(self, input_shape):
        self.rnn_cell = tf.keras.layers.LSTM(units=self.hid_dim, return_state=True)
        self.temp_decay_h = TemporalDecay(units=self.hid_dim, diag=False)
        self.temp_decay_x = TemporalDecay(units=self.internal_dim, diag=True)
        self.hist_reg = Dense(units=self.internal_dim, activation='linear')
        self.feat_reg = FeatureRegression()
        self.weight_combine = Dense(units=self.internal_dim, activation='linear')
        self.dropout = Dropout(rate=0.25)
        self.out = Dense(units=1, activation='linear')
        self.sequence_length = input_shape[1]

    # @tf.function
    def call(self, values, masks, deltas):
        h = tf.zeros(shape=(values.shape[0], self.hid_dim))
        c = tf.zeros(shape=(values.shape[0], self.hid_dim))
        ### IMPLEMENT CUSTOM LOSSES ###
        custom_loss_x = 0.0

        imputations = []
        for t in range(self.sequence_length):
            x = values[:, t, :]
            m = masks[:, t, :]
            d = deltas[:, t, :]

            gamma_h = self.temp_decay_h(d)
            gamma_x = self.temp_decay_x(d)

            h = h * gamma_h

            x_h = self.hist_reg(h)
            custom_loss_x += tf.reduce_sum((tf.abs(x - x_h) * m) / (tf.reduce_sum(m) + 1e-6))
            x_c = m * x + (1 - m) * x_h

            z_h = self.feat_reg(x_c)

            alpha = self.weight_combine(tf.concat([gamma_x, m], axis=1))

            c_h = alpha * z_h + (1 - alpha) * x_h
            custom_loss_x += tf.reduce_sum((tf.abs(x - c_h) * m) / (tf.reduce_sum(m) + 1e-6))
            c_c = m * x + (1 - m) * c_h
            imputations.append(c_c)
            inputs = tf.concat([c_c, m], axis=1)

            _, h, c = self.rnn_cell(tf.expand_dims(inputs, axis=1), [h, c])

        imputations = tf.concat([tf.expand_dims(f, axis=1) for f in imputations], axis=1)
        y_h = self.out(h)
        return y_h, imputations


debug = RITS(internal_dim=5, hid_dim=10, sequence_length=3)

print("Debug Ready")
x = tf.ones(shape=(256, 3, 5))
m = tf.ones(shape=(256, 3, 5))
d = tf.zeros(shape=(256, 3, 5))

debug(x, m, d)
