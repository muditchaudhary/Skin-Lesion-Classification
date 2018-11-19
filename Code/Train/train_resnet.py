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
<<<<<<< HEAD
from tensorflow import keras

print(tf.__version__)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.001, patience=10)
csv_logger = CSVLogger('resnet18_ISIC.csv')


featdef = {
    'train/image': FixedLenFeature(shape=[], dtype= tf.string),
    'train/label': FixedLenFeature(shape = [], dtype = tf.int64)
}

def _parse_record (example_proto, clip = False):
    ex = tf.parse_single_example(example_proto, featdef)
    
    im = tf.decode_raw(ex['train/image'], tf.uint8)
    im = tf.reshape(im, (28,28,3))
    im = tf.cast(im, tf.float32)*(1./255)
        
    label = (ex['train/label'])
    label = tf.keras.utils.to_categorical(ex['train/label'], 2)
    print(label)
    print(im)
    return im, label

batch_size = 20
ds_train = tf.data.TFRecordDataset('/home/mudit/Skin Lesion Classification/TFrecord_Datasets/Melanoma_training_uint8.tfrecords').map(_parse_record)

ds_validation = tf.data.TFRecordDataset('/home/mudit/Skin Lesion Classification/TFrecord_Datasets/Melanoma_validation_uint8.tfrecords').map(_parse_record)

ds_train = ds_train.repeat().batch(batch_size)
ds_validation = ds_validation.repeat().batch(150)
ds_validation = ds_validation.repeat().batch(batch_size)

tensor = ds_train.make_one_shot_iterator().get_next()
with tf.Session() as session:
    print(session.run(tensor))

IM_SIZE = 28
nb_classes = 2
image_input = tf.keras.Input(shape = (IM_SIZE, IM_SIZE, 3), name = 'input_layer')


model = resnet.ResnetBuilder.build_resnet_18((3, 28, 28), nb_classes)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
<<<<<<< HEAD
=======

print(model.summary())
>>>>>>> c2e9d39f27e12a8d028b142ed4224633a2137cc6

print(model.summary())

<<<<<<< HEAD
weight=[1,5]
#tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,  write_graph=True, write_images=False)

print('Not using data augmentation.')
history = model.fit(ds_train, batch_size=None, epochs=2, shuffle=True, validation_data = ds_validation, steps_per_epoch=100, validation_steps = 1,class_weight=weight)
=======
#tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,  write_graph=True, write_images=False)

print('Not using data augmentation.')
history = model.fit(ds_train, batch_size=None, epochs=2, shuffle=True, validation_data = ds_validation, steps_per_epoch=100, validation_steps = 7)
>>>>>>> c2e9d39f27e12a8d028b142ed4224633a2137cc6

print(history.history.keys())


# In[79]:


#model.predict(ds_train, steps =1)

file_pi=open('trainHistoryDict', "wb")
pickle.dump(history.history,file_pi)
file_pi.close()

plt.plot(history.history['acc'])
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




