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
from tensorflow import keras
from tensorboard.plugins.pr_curve import summary as pr_summary
import keras.backend as K
from itertools import product
from functools import partial


print(tf.__version__)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.001, patience=10)
csv_logger = CSVLogger('resnet18_ISIC.csv')

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


def create_dataset(filepath, batch_size, shuffle):
    dataset = tf.data.TFRecordDataset(filepath)

    dataset = dataset.map(_parse_function,num_parallel_calls=8)

    dataset = dataset.repeat()

    if shuffle is True:
        dataset = dataset.shuffle(100)

    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()

    image, label = iterator.get_next()

    image = tf.reshape(image, [-1,224,224,3])
    image = tf.cast(image, tf.float32)*(1./255)
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

trainImage, trainLabel = create_dataset('/home/mudit/Skin Lesion Classification/TFrecord_Datasets/Melanoma_training_224_uint8.tfrecords', 25, True)
valImage, valLabel = create_dataset('/home/mudit/Skin Lesion Classification/TFrecord_Datasets/Melanoma_validation_FalseShuffle_224_uint8.tfrecords',150, False)

Test = np.genfromtxt('ISIC-2017_Validation_Part3_GroundTruth.csv', delimiter=',', usecols=(1), skip_header=1)
Test  = Test.tolist()

Test = [1 if i == 1.0 else 0 for i in Test]

print("TEST", Test)
print("\nVALD")


#run the graph
with tf.Session() as sess:
    print (sess.run(valLabel))


IMSIZE = 224
nb_classes = 2

w_array = np.ones((2,2))
w_array[1,0] = 5
w_array[0,1] = 1

ncce = partial(w_categorical_crossentropy,weights=w_array)

model = resnet.ResnetBuilder.build_resnet_18((3, IMSIZE, IMSIZE), nb_classes)

model.compile(loss=ncce,
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())

weight=[1,5]


print('Not using data augmentation.')
history = model.fit(x=trainImage,y=trainLabel, batch_size=None, epochs=1, shuffle=True, validation_data = (valImage,valLabel), steps_per_epoch=80, validation_steps = 1, class_weight=weight, verbose=1)


print(history.history.keys())

Preds = model.predict(valImage, steps = 1)
Preds = np.argmax(Preds, axis = 1)
print(Preds)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Test, Preds)

print(cm)


file_pi=open('trainHistoryDict', "wb")
pickle.dump(history.history,file_pi)
file_pi.close()

"""plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc = 'upper left')
plt.show()


# In[35]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc = 'upper left')
plt.show()
"""
