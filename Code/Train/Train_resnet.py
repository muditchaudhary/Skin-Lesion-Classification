"""
To fine tune ResNet-50
"""
from __future__ import print_function

#To reduce verbosity
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

#For FineTuning ResNet-50
from tensorflow.keras.applications.resnet50 import ResNet50

print(tf.__version__)

#Address for positive only dataset
addPosPath = 'Imagenet/Seb_Training_Augmented_AddPos_Imagenet.tfrecords'

def lr_scheduler(epoch,lr):
    """
    Learning rate scheduler decays the learning rate by factor of 0.1 every 10 epochs after 20 epochs
    """
    decay_rate = 0.1
    if epoch==20:
        return lr*decay_rate
    elif epoch%10==0 and epoch >20:
        return lr*decay_rate
    return lr

LRScheduler = keras.callbacks.LearningRateScheduler(lr_scheduler,verbose=1)

class Metrics(keras.callbacks.Callback):
    """
    Implementation of custom metrics: Precision, Recall, F-Measure and Confusion Matrix
    """
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
    """
    Parser for TFRecord file
    """
    keys_to_features = {'train/image': tf.FixedLenFeature([], tf.string),
    'train/label': tf.FixedLenFeature([], tf.int64)}

    parsed_features = tf.parse_single_example(proto, keys_to_features)
    parsed_features['train/image'] = tf.decode_raw(parsed_features['train/image'], tf.float32)

    return parsed_features['train/image'], parsed_features["train/label"]


def create_dataset(filepath, batch_size, shuffle, augmentfilepath, augment, addPosPath, addPos):
    """
    Reads TFRecord and creates the dataset. Returns image and label dataset as tensors.
    """
    dataset = tf.data.TFRecordDataset(filepath)

    #If want to add augmented dataset, put augmentfilepath
    if augment is True:
        augmented = tf.data.TFRecordDataset(augmentfilepath)
        dataset = dataset.concatenate(augmented)
    
    #If want to add positive only dataset, put addPosPath
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

    #Image reshaped to 224x224x3 to match ImageNet dataset
    image = tf.reshape(image, [-1,224,224,3])
    image = tf.cast(image, tf.float32)
    label = tf.one_hot(label, 2)

    return image, label


def w_categorical_crossentropy(y_true, y_pred, weights):
    """
    Implementation of Weighted Categorical Crossentropy Function for unbalanced datasets 
    """
    nb_cl = len(weights)
    final_mask = K.zeros_like(y_pred[:, 0])
    y_pred_max = K.max(y_pred, axis=1)
    y_pred_max = K.expand_dims(y_pred_max, 1)
    y_pred_max_mat = K.equal(y_pred, y_pred_max)
    for c_p, c_t in product(range(nb_cl), range(nb_cl)):

        final_mask += (K.cast(weights[c_t, c_p],K.floatx()) * K.cast(y_pred_max_mat[:, c_p] ,K.floatx())* K.cast(y_true[:, c_t],K.floatx()))
    return K.categorical_crossentropy(y_true, y_pred) * final_mask

trainImage, trainLabel = create_dataset('Imagenet/Seb_Training_Imagenet.tfrecords', 32, True, 'Imagenet/Seb_Training_Augmented_Imagenet.tfrecords', True, addPosPath, True)
valImage, valLabel = create_dataset('Imagenet/Seb_Test_Imagenet.tfrecords',600, False,'',False, '', False)

Test = np.genfromtxt('ISIC-2017_Validation_Part3_GroundTruth.csv', delimiter=',', usecols=(1), skip_header=1)
Test  = Test.tolist()

Test = [1 if i == 1.0 else 0 for i in Test]
    
IMSIZE = 224
nb_classes = 2

w_array = np.ones((2,2))

#Weights for weighted loss function
w_array[1,0] = 6
w_array[0,1] = 1
print(w_array)

#For weight checkpoint
checkpoint = ModelCheckpoint('weights_resnetb{epoch:03d}.h5',save_weights_only = True, period = 1)
ncce = partial(w_categorical_crossentropy,weights=w_array)
metrics = Metrics()

model = resnet.ResnetBuilder.build_resnet_101((3, IMSIZE, IMSIZE), nb_classes)

model.compile(loss=ncce,
              optimizer=keras.optimizers.SGD(lr=0.01,momentum =0.9),
              metrics=['accuracy'])

print(model.summary())


print('Using data augmentation.')
history = model.fit(x=trainImage,y=trainLabel, batch_size=None, epochs=50, shuffle=True, validation_data = (valImage,valLabel), steps_per_epoch=132, validation_steps = 1,  verbose=1, callbacks=[metrics, checkpoint,LRScheduler])

metricsData = metrics.get_data()
file_pi=open('trainHistoryDict', "wb")
pickle.dump(history.history,file_pi)
file_pi.close()

file_pi = open('metricsDict',"wb")
pickle.dump(metricsData, file_pi)
file_pi.close()
