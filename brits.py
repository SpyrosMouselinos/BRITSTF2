import tensorflow as tf
from tensorflow.keras.models import Model
from rits import RITS
import numpy as np


class BRITS(Model):
    def __init__(self, internal_dim, hid_dim, sequence_length=None, name='BRITS'):
        super(BRITS, self).__init__(name=name)
        self.hid_dim = hid_dim
        self.internal_dim = internal_dim
        self.sequence_length = sequence_length
        return

    def build(self, input_shape):
        """Build the RITS layers for the model.

        This method initializes two RITS layers, one for processing the input
        sequence in the forward direction and another for processing it in the
        backward direction. The sequence length is derived from the input shape
        provided. The RITS layers are configured with specified internal and
        hidden dimensions.

        Args:
            input_shape (tuple): A tuple representing the shape of the input data,
                where the second element indicates the sequence length.
        """

        self.sequence_length = input_shape[1]
        self.rits_f = RITS(self.internal_dim, self.hid_dim, self.sequence_length, go_backwards=False, name="RITS_F")
        self.rits_b = RITS(self.internal_dim, self.hid_dim, self.sequence_length, go_backwards=True, name="RITS_B")
        return

    def call(self, values, masks, deltas):
        """Call the forward and backward RITS functions to compute predictions and
        custom loss.

        This method takes input values, masks, and deltas, and computes
        predictions using both forward and backward RITS functions. It averages
        the predictions and custom loss from both directions to provide a final
        output. The method also returns the imputations from both forward and
        backward passes.

        Args:
            values (array-like): The input values for the RITS functions.
            masks (array-like): The masks to be applied to the input values.
            deltas (array-like): The deltas used in the RITS computations.

        Returns:
            tuple: A tuple containing:
                - predictions (array-like): The averaged predictions from both forward
                and backward RITS.
                - list: A list containing imputations from both forward and backward
                RITS.
                - custom_loss (float): The averaged custom loss from both forward and
                backward RITS.
        """

        predictions_f, imputations_f, custom_loss_f = self.rits_f(values, masks, deltas)
        predictions_b, imputations_b, custom_loss_b = self.rits_b(values, masks, deltas)
        predictions = (predictions_f + predictions_b) / 2.0
        custom_loss = (custom_loss_f + custom_loss_b) / 2.0
        return predictions, [imputations_f, imputations_b], custom_loss


brit_model = BRITS(internal_dim=3, hid_dim=5)
opt = tf.keras.optimizers.Adam(1e-3)

for i in range(0, 50):
    x = np.random.uniform(low=0, high=1, size=(256, 10, 3)).astype('float32')
    m = np.random.randint(low=0, high=2, size=(256, 10, 3)).astype('float32')
    d = np.random.randint(low=0, high=2, size=(256, 10, 3)).astype('float32')
    y = x / 2.0
    y = y[:, :, 0]
    with tf.GradientTape() as tape:
        predictions, imputations, custom_loss = brit_model(x, m, d)
        prediction_loss = tf.keras.losses.mean_squared_error(y, predictions)
        discrepancy_loss = tf.reduce_mean(tf.keras.losses.mean_absolute_error(imputations[0], imputations[1]), axis=1)
        loss = prediction_loss + discrepancy_loss + custom_loss
    gradients = tape.gradient(loss, brit_model.trainable_variables)
    opt.apply_gradients(zip(gradients, brit_model.trainable_variables))
    tf.print(loss.numpy().mean())

p, i, _ = brit_model(x[0:1], m[0:1], d[0:1])
print(p)
print(i)
