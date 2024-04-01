
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Layer
from tensorflow.keras.regularizers import l2

from rec.layers import FFM_Layer


class FFM(Model):
    def __init__(self, feature_columns, k=8, w_reg=0., v_reg=0.):
        """Field-aware Factorization Machines.
        Args:
            :param feature_columns: A list. [{'feat_name':, 'feat_num':, 'embed_dim':}, ...]
            :param k: A scalar. The latent vector.
            :param w_reg: A scalar. The regularization coefficient of parameter w.
            :param v_reg: A scalar. The regularization coefficient of parameter v.
        :return:
        """
        super(FFM, self).__init__()
        self.feature_columns = feature_columns
        self.ffm = FFM_Layer(self.feature_columns, k, w_reg, v_reg)

    def call(self, inputs):
        ffm_out = self.ffm(inputs)
        outputs = tf.nn.sigmoid(ffm_out)
        return outputs

    def summary(self):
        inputs = {
            feat['feat_name']: Input(shape=(), dtype=tf.int32, name=feat['feat_name'])
            for feat in self.feature_columns
        }
        Model(inputs=inputs, outputs=self.call(inputs)).summary()
