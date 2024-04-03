import tensorflow as tf
from keras.losses import binary_crossentropy
from keras.optimizers.legacy import Adam
from keras.metrics import AUC

from rec.models.ranking import FM
from rec.data.datasets.criteo import create_small_criteo_dataset

import pickle
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# If you have GPU, and the value is GPU serial number.
os.environ['CUDA_VISIBLE_DEVICES'] = '6'


learning_rate = 0.001
batch_size = 4096
model_params = {
    'k': 8,
    'w_reg': 0.,
    'v_reg': 0.
}


def easy_demo(file, sample_num=500000, read_part=True, test_size=0.1, epochs=10):
    feature_columns, train, test = create_small_criteo_dataset(file=file,
                                                         read_part=read_part,
                                                         sample_num=sample_num,
                                                         test_size=test_size)
    train_X, train_y = train
    test_X, test_y = test

    model = FM(feature_columns=feature_columns, **model_params)
    # model.summary()
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate),
                  metrics=[AUC()])

    model.fit(
        train_X,
        train_y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1
    )

    print('test AUC: %f' % model.evaluate(test_X, test_y, batch_size=batch_size)[1])





if __name__ == '__main__':
    file = 'data/criteo/train.txt'
    easy_demo(file)
