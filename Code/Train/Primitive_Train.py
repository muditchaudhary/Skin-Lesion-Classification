#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


from tensorflow import FixedLenFeature


# In[4]:


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
    label = tf.one_hot(ex['train/label'], 2)
    print(label)
    print(im)
    return im, label
    
batch_size = 20
ds_train = tf.data.TFRecordDataset('/home/mudit/Skin Lesion Classification/TFrecord_Datasets/Melanoma_training_uint8.tfrecords').map(_parse_record)

ds_validation = tf.data.TFRecordDataset('/home/mudit/Skin Lesion Classification/TFrecord_Datasets/Melanoma_validation_uint8.tfrecords').map(_parse_record)

ds_train = ds_train.repeat().batch(batch_size)
ds_validation = ds_validation.repeat().batch(150)


# In[82]:


tensor = ds_train.make_one_shot_iterator().get_next()
with tf.Session() as session:
    print(session.run(tensor))


# In[67]:


IM_SIZE = 28

image_input = tf.keras.Input(shape = (IM_SIZE, IM_SIZE, 3), name = 'input_layer')


# In[68]:


conv_1 = tf.keras.layers.Conv2D(32, kernel_size = (3,3),
                              padding = 'same',
                              activation = 'relu')(image_input)


# In[69]:


conv1 = tf.keras.layers.MaxPooling2D(padding = 'same')(conv_1)


# In[70]:


conv_2 = tf.keras.layers.Conv2D(32, kernel_size = (3,3), padding = 'same',
                               activation = 'relu')(conv_1)


# In[71]:


conv_flat = tf.keras.layers.Flatten()(conv_2)


# In[72]:


fc_1 = tf.keras.layers.Dense(28,
                             activation='relu')(conv_flat)
fc_1 = tf.keras.layers.Dropout(0.2)(fc_1)
fc_2 = tf.keras.layers.Dense(28,
                             activation='relu')(fc_1)
fc_2 = tf.keras.layers.Dropout(0.4)(fc_2)


# In[73]:


label_output = tf.keras.layers.Dense(2,
                                       activation='softmax',
                                       name='label')(fc_2)


# In[74]:


model = tf.keras.Model(inputs = image_input, outputs = [label_output])


# In[75]:


print(model.summary())


# In[76]:


model.compile(optimizer = 'sgd', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# In[77]:


history = model.fit(ds_train, steps_per_epoch = 100, epochs = 5, validation_data =ds_validation, validation_steps=1)


# In[78]:

print(model.evaluate(ds_validation, steps=1))
print(history.history.keys())


# In[79]:


predictions = model.predict(ds_validation, steps =1)

for i in range(len(predictions)):
    print(predictions[i])

# In[21]:
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

