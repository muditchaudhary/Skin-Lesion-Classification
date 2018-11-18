from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, TensorBoard
import resnet
from time import time
import pickle
import tensorflow as tf
import numpy as np
from tensorflow import FixedLenFeature
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorboard.plugins.pr_curve import summary as pr_summary
import keras.backend as K
from itertools import product
from functools import partial


print(tf.__version__)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.001, patience=10)
csv_logger = CSVLogger('resnet18_ISIC.csv')
augmentation = False

def precision(y_true, y_pred):
    """Precision metric.
     Only computes a batch-wise average of precision.
     Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    return precision

def _parse_function(proto):
    keys_to_features = {'train/image': tf.FixedLenFeature([], tf.string),
    'train/label': tf.FixedLenFeature([], tf.int64)}

    parsed_features = tf.parse_single_example(proto, keys_to_features)
    parsed_features['train/image'] = tf.decode_raw(parsed_features['train/image'], tf.uint8)

    return parsed_features['train/image'], parsed_features["train/label"]


def create_dataset(filepath, batch_size):
    dataset = tf.data.TFRecordDataset(filepath)

    dataset = dataset.map(_parse_function,num_parallel_calls=8)

    dataset = dataset.repeat()

    dataset = dataset.shuffle(10)

    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()

    image, label = iterator.get_next()

    image = tf.reshape(image, [-1,28,28,3])
    image = tf.cast(image, tf.float32)*(1./255)

    label = tf.one_hot(label, 2)

    

    print("batch")
    

    return image, label


def w_categorical_crossentropy(y_true, y_pred, weights):
    nb_cl = len(weights)
    final_mask = K.zeros_like(y_pred[:, 0])
    y_pred_max = K.max(y_pred, axis=1)
    y_pred_max = K.expand_dims(y_pred_max, 1)
    y_pred_max_mat = K.equal(y_pred, y_pred_max)
    for c_p, c_t in product(range(nb_cl), range(nb_cl)):

        final_mask += (K.cast(weights[c_t, c_p],K.floatx()) * K.cast(y_pred_max_mat[:, c_p] ,K.floatx())* K.cast(y_true[:, c_t],K.floatx()))
    return K.categorical_crossentropy(y_pred, y_true) * final_mask


trainImage, trainLabel = create_dataset('/home/mudit/Skin Lesion Classification/TFrecord_Datasets/melanoma_training_244_uint8.tfrecords', 25)
valImage, valLabel = create_dataset('/home/mudit/Skin Lesion Classification/TFrecord_Datasets/Melanoma_validation_244_uint8.tfrecords',150)

print("\n\nIMAGETYPE ",trainImage.shape)
IM_SIZE = 28
nb_classes = 2

w_array = np.ones((2,2))
w_array[1,0] = 5
w_array[0,1] = 5

ncce = partial(w_categorical_crossentropy,weights=w_array)

IM_SIZE = 224
nb_classes = 2

model = resnet.ResnetBuilder.build_resnet_18((3, IM_SIZE, IM_SIZE), nb_classes)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())

if not augmentation:
    print('Not using data augmentation.')
    history = model.fit(x=trainImage,y=trainLabel, batch_size=None, epochs=6, shuffle=True, validation_data = (valImage,valLabel), steps_per_epoch=80, validation_steps = 1, class_weight=weight, verbose=1)

else:
    print('Using real-time data augmentation')

    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=90,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images

    datagen.fit(trainImage)

    model.fit_generator(datagen.flow(x=trainImage,y=trainLabel, batch_size=None, epochs=6, shuffle=True, validation_data = (valImage,valLabel), steps_per_epoch=80, validation_steps = 1,verbose=1, max_q_size=100))

print(history.history.keys())

file_pi=open('trainHistoryDict', "wb")
pickle.dump(history.history,file_pi)
file_pi.close()
