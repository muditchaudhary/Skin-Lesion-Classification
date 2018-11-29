from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']= '3'

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, TensorBoard
import resnet
from time import time
import pickle
import tensorflow as tf
import numpy as np
from tensorflow import FixedLenFeature
from tensorflow import keras
import keras.backend as K
from itertools import product
from functools import partial
import sklearn.metrics
from keras.callbacks import ModelCheckpoint

from tensorflow.keras.applications.resnet50 import ResNet50

print(tf.__version__)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.001, patience=10)
csv_logger = CSVLogger('resnet18_ISIC.csv')
addPosPath = 'Imagenet/Melanoma_Training_Augmented_AddPos_Imagenet.tfrecords'

def lr_scheduler(epoch,lr):
    decay_rate = 0.1
    decay_step = 20
    if epoch % decay_step == 0 and epoch:
        return lr*decay_rate
    return lr

LRScheduler = keras.callbacks.LearningRateScheduler(lr_scheduler,verbose=1)

class Metrics(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self._data = []

    def on_epoch_end(self, batch, logs={}):
        X_val, y_val = self.validation_data[0], self.validation_data[1]
        y_predict = np.asarray(model.predict(X_val, steps = 1))
        
        with tf.Session() as sess:
            y_val = (sess.run(y_val))

        y_val = np.argmax(y_val, axis=1)
        y_predict = np.argmax(y_predict, axis=1)
        print("\nMetrics for Epoch")
        print("Confusion Matrix:\n",sklearn.metrics.confusion_matrix(y_val,y_predict))
        print("Recall: ", sklearn.metrics.recall_score(y_val,y_predict, pos_label = 1))
        print("Precision: ", sklearn.metrics.precision_score(y_val,y_predict, pos_label=1))
        print("F1_score: ", sklearn.metrics.f1_score(y_val,y_predict,pos_label=1))
        print("ROC_AUC_score: ", sklearn.metrics.roc_auc_score(y_val,y_predict))
        print("\n") 
        self._data.append({
            'val_recall': sklearn.metrics.recall_score(y_val, y_predict, pos_label=1),
            'val_precision': sklearn.metrics.precision_score(y_val, y_predict, pos_label=1),
            'val_f1_score': sklearn.metrics.f1_score(y_val,y_predict, pos_label = 1),
            'val_roc_auc_score': sklearn.metrics.roc_auc_score(y_val,y_predict)
        })
        return

    def get_data(self):
        return self._data



def _parse_function(proto):
    keys_to_features = {'train/image': tf.FixedLenFeature([], tf.string),
    'train/label': tf.FixedLenFeature([], tf.int64)}

    parsed_features = tf.parse_single_example(proto, keys_to_features)
    parsed_features['train/image'] = tf.decode_raw(parsed_features['train/image'], tf.float32)

    return parsed_features['train/image'], parsed_features["train/label"]


def create_dataset(filepath, batch_size, shuffle, augmentfilepath, augment, addPosPath, addPos):
    dataset = tf.data.TFRecordDataset(filepath)

    if augment is True:
        augmented = tf.data.TFRecordDataset(augmentfilepath)
        dataset = dataset.concatenate(augmented)
    
    if addPos is True:
        added = tf.data.TFRecordDataset(addPosPath)
        dataset = dataset.concatenate(added)

    dataset = dataset.map(_parse_function,num_parallel_calls=8)

    dataset = dataset.repeat()

    if shuffle is True:
        dataset = dataset.shuffle(5000)
        dataset = dataset.shuffle(800)

    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()

    image, label = iterator.get_next()

    image = tf.reshape(image, [-1,224,224,3])
    image = tf.cast(image, tf.float32)
    label = tf.one_hot(label, 2)

    return image, label


def w_categorical_crossentropy(y_true, y_pred, weights):
    nb_cl = len(weights)
    final_mask = K.zeros_like(y_pred[:, 0])
    y_pred_max = K.max(y_pred, axis=1)
    y_pred_max = K.expand_dims(y_pred_max, 1)
    y_pred_max_mat = K.equal(y_pred, y_pred_max)
    for c_p, c_t in product(range(nb_cl), range(nb_cl)):

        final_mask += (K.cast(weights[c_t, c_p],K.floatx()) * K.cast(y_pred_max_mat[:, c_p] ,K.floatx())* K.cast(y_true[:, c_t],K.floatx()))
    return K.categorical_crossentropy(y_true, y_pred) * final_mask

ValImage, ValLabel = create_dataset('/home/mudit/Skin Lesion Classification/TFrecord_Datasets/Imagenet/Melanoma_Validation_Imagenet.tfrecords',150, False,'',False, '', False)

TestImage, TestLabel = create_dataset('/home/mudit/Skin Lesion Classification/TFrecord_Datasets/Imagenet/Melanoma_Test_Imagenet.tfrecords',150, False,'',False, '', False)

TestCSV = np.genfromtxt('ISIC-2017_Validation_Part3_GroundTruth.csv', delimiter=',', usecols=(1), skip_header=1)
TestCSV  = TestCSV.tolist()

TestCSV = [1 if i == 1.0 else 0 for i in TestCSV]
    
IMSIZE = 224
nb_classes = 2

check=[]

with tf.Session() as sess:
    check= (sess.run(ValLabel).tolist())

print(check)
print("\n\n")
print(TestCSV)
w_array = np.ones((2,2))
w_array[1,0] = 6
w_array[0,1] = 1

print(w_array)


checkpoint = ModelCheckpoint('weights_resnet{epoch:03d}.h5',save_weights_only = True, period = 1)
ncce = partial(w_categorical_crossentropy,weights=w_array)
metrics = Metrics()

input_tensor = keras.layers.Input(shape = (224,224,3))


model = ResNet50(input_tensor= input_tensor,weights='imagenet',include_top=False)

for layer in model.layers[:-33]:
    layer.trainable = False

x = model.output
x= tf.keras.layers.GlobalAveragePooling2D(data_format='channels_last')(x)
x=tf.keras.layers.Dense(nb_classes,activation ='softmax')(x)

model = tf.keras.models.Model(model.input, x)

model.compile(loss=ncce,
              optimizer=tf.keras.optimizers.SGD(lr=0.0001, momentum =0.9),
              metrics=['accuracy'])

print(model.summary())
model.load_weights('weights_resnet012.h5')
Preds = np.asarray(model.predict(ValImage, steps = 1))
Preds = np.argmax(Preds, axis = 1)
print(len(Preds))
print("F1_score: ", sklearn.metrics.f1_score(TestCSV,Preds,pos_label=1))
print("ROC_AUC_score: ", sklearn.metrics.roc_auc_score(TestCSV,Preds))


