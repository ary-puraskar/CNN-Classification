# # Convolutional Neural Network

# ### Importing the libraries


get_ipython().system('pip uninstall -y tensorflow tensorflow-cpu tensorflow-intel keras ml-dtypes tensorboard')


get_ipython().system('pip install tensorflow-intel==2.15.0                keras==2.15.0                ml-dtypes==0.2.0                tensorboard==2.15.0')


import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator 


# ## Part 1 - Data Preprocessing

# ### Preprocessing the Training set


import os
os.getcwd()


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')


# ### Preprocessing the Test set


test_datagen=ImageDataGenerator(rescale=1./255)
test_set=test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')


# ## Part 2 - Building the CNN

# ### Initialising the CNN

dl=tf.keras.models.Sequential()


# ### Step 1 - Convolution

dl.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu',input_shape=[64,64,3]))


# ### Step 2 - Pooling


dl.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))


### Adding a second convolutional layer

dl.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu'))
dl.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))


# ### Step 3 - Flattening


dl.add(tf.keras.layers.Flatten())


# ### Step 4 - Full Connection


dl.add(tf.keras.layers.Dense(units=128, activation='relu'))


# ### Step 5 - Output Layer


dl.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


# ## Part 3 - Training the CNN

# ### Compiling the CNN


dl.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# ### Training the CNN on the Training set and evaluating it on the Test set


dl.fit(x = training_set, validation_data =test_set ,epochs= 25)


# ## Part 4 - Making a single prediction

import numpy as np
from keras.preprocessing import image

# Load and preprocess the image
test_image = image.load_img('dataset/single_prediction/cat_or_dog.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
test_image = test_image / 255.0  # Add this line for rescaling

# Make prediction
result = dl.predict(test_image)
print("Raw prediction value:", result[0][0])  # Add this to see actual output

# Get class indices 
class_indices = training_set.class_indices
print("Class indices:", class_indices)  # Verify which class is 0 and which is 1

# Assuming cats: 0, dogs: 1
if result[0][0] > 0.5:  # Changed from == 1 to > 0.5
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)

