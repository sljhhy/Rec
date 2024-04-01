import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout, BatchNormalization, LayerNormalization, Conv1D, ReLU
from tensorflow.keras.regularizers import l2

from rec.layers.utils import scaled_dot_product_attention, split_heads, index_mapping

class FM_Layer(Layer):
    def __init__(self, feature_columns, k, w_reg=0., v_reg=0.):
        """Factorization Machines.
        Args:
            :param feature_columns: A list. [{'feat_name':, 'feat_num':, 'embed_dim':}, ...]
            :param k: A scalar. The latent vector.
            :param w_reg: A scalar. The regularization coefficient of parameter w.
            :param v_reg: A scalar. The regularization coefficient of parameter v.
        :return:
        """
        super(FM_Layer, self).__init__()
        self.feature_columns = feature_columns
        self.field_num = len(feature_columns)
        self.map_dict = {}
        self.feature_length = 0
        for feat in self.feature_columns:
            self.map_dict[feat['feat_name']] = self.feature_length
            self.feature_length += feat['feat_num']
        self.k = k
        self.w_reg = w_reg
        self.v_reg = v_reg

    def build(self, input_shape):
        self.w0 = self.add_weight(name='w0', shape=(1,),
                                  initializer=tf.zeros_initializer(),
                                  trainable=True)
        self.w = self.add_weight(name='w', shape=(self.feature_length, 1),
                                 initializer=tf.random_normal_initializer(),
                                 regularizer=l2(self.w_reg),
                                 trainable=True)
        self.V = self.add_weight(name='V', shape=(self.feature_length, self.k),
                                 initializer=tf.random_normal_initializer(),
                                 regularizer=l2(self.v_reg),
                                 trainable=True)

    def call(self, inputs):
        # mapping
        inputs = index_mapping(inputs, self.map_dict)
        inputs = tf.concat([value for _, value in inputs.items()], axis=-1)
        # first order
        first_order = self.w0 + tf.reduce_sum(tf.nn.embedding_lookup(self.w, inputs), axis=1)  # (batch_size, 1)
        # second order
        second_inputs = tf.nn.embedding_lookup(self.V, inputs)  # (batch_size, fields, embed_dim)
        square_sum = tf.square(tf.reduce_sum(second_inputs, axis=1, keepdims=True))  # (batch_size, 1, embed_dim)
        sum_square = tf.reduce_sum(tf.square(second_inputs), axis=1, keepdims=True)  # (batch_size, 1, embed_dim)
        second_order = 0.5 * tf.reduce_sum(square_sum - sum_square, axis=2)  # (batch_size, 1)
        # outputs
        outputs = first_order + second_order
        return outputs


class FFM_Layer(Layer):
    def __init__(self, feature_columns, k, w_reg=0., v_reg=0.):
        """Field Factorization Machines Layer.
        Args:
            :param feature_columns: A list. [{'feat_name':, 'feat_num':, 'embed_dim':}, ...]
            :param k: A scalar. The latent vector.
            :param w_reg: A scalar. The regularization coefficient of parameter w.
            :param v_reg: A scalar. The regularization coefficient of parameter v.
        :return:
        """
        super(FFM_Layer, self).__init__()
        self.feature_columns = feature_columns
        self.field_num = len(self.feature_columns)
        self.k = k
        self.w_reg = w_reg
        self.v_reg = v_reg
        self.map_dict = {}
        self.feature_length = 0
        for feat in self.feature_columns:
            self.map_dict[feat['feat_name']] = self.feature_length
            self.feature_length += feat['feat_num']

    def build(self, input_shape):
        self.w0 = self.add_weight(name='w0', shape=(1,),
                                  initializer=tf.zeros_initializer(),
                                  trainable=True)
        self.w = self.add_weight(name='w', shape=(self.feature_length, 1),
                                 initializer='random_normal',
                                 regularizer=l2(self.w_reg),
                                 trainable=True)
        self.v = self.add_weight(name='v',
                                 shape=(self.feature_length, self.field_num, self.k),
                                 initializer='random_normal',
                                 regularizer=l2(self.v_reg),
                                 trainable=True)

    def call(self, inputs):
        inputs = index_mapping(inputs, self.map_dict)
        inputs = tf.concat([value for _, value in inputs.items()], axis=-1)
        # first order
        first_order = self.w0 + tf.reduce_sum(tf.nn.embedding_lookup(self.w, inputs), axis=1)  # (batch_size, 1)
        # field second order
        second_order = 0
        latent_vector = tf.reduce_sum(tf.nn.embedding_lookup(self.v, inputs), axis=1)  # (batch_size, field_num, k)
        for i in range(self.field_num):
            for j in range(i + 1, self.field_num):
                second_order += tf.reduce_sum(latent_vector[:, i] * latent_vector[:, j], axis=1, keepdims=True)
        return first_order + second_order
