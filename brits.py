import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.layers import Layer
from rits import RITS


class BRITS(Model):
    def __init__(self, internal_dim, hid_dim, sequence_length=None, name='BRITS'):
        super(BRITS, self).__init__(name=name)
        self.hid_dim = hid_dim
        self.internal_dim = internal_dim
        self.sequence_length = sequence_length
        return

    def build(self, input_shape):
        self.sequence_length = input_shape[1]
        self.rits_f = RITS(self.internal_dim, self.hid_dim, self.sequence_length, go_backwards=False, name="RITS_F")
        self.rits_b = RITS(self.internal_dim, self.hid_dim, self.sequence_length, go_backwards=True, name="RITS_B")
        return

    def call(self, values, masks, deltas):
        loss_f, predictions_f, imputations_f = self.rits_f(values, masks, deltas)
        loss_b, predictions_b, imputations_b = self.rits_b(values, masks, deltas)
        loss = loss_f + loss_b
        return loss


brit_model = BRITS(internal_dim=5, hid_dim=10, sequence_length=3)

print("Debug Ready")
x = tf.ones(shape=(1, 3, 5))
m = tf.ones(shape=(1, 3, 5))
d = tf.zeros(shape=(1, 3, 5))
#y = tf.ones(shape=(1, 3)) * 0.5
opt = tf.keras.optimizers.Adam()

for i in range(0, 20):
    with tf.GradientTape() as tape:
        loss = brit_model(x, m, d)
    gradients = tape.gradient(loss, brit_model.trainable_variables)
    opt.apply_gradients(zip(gradients, brit_model.trainable_variables))
    tf.print(loss)
