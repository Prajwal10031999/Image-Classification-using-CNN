from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from PIL import Image


# Sequential NN
classifier = Sequential()

# Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Flatten
classifier.add(Flatten())

# Fully Connected
classifier.add(Dense(units = 128, activation = 'relu'))

# Output Layer
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Model Compilation
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', 
metrics = ['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('Dataset/train',
target_size = (64, 64),
batch_size = 32,
class_mode = 'binary')
test_set = test_datagen.flow_from_directory('Dataset/test/',
target_size = (64, 64),
batch_size = 32,
class_mode = 'binary')


classifier.fit_generator(training_set,
steps_per_epoch = 109,
epochs = 6,
validation_data = test_set, validation_steps = 36)


import numpy as np
from IPython.display import Image,display
# To display the image in jupyter notebook

from keras.preprocessing import image

def who(img_file):
    img_name = img_file
# Image Pre-processing

    test_image = image.load_img(img_name, target_size = (64, 64))
# displaying image

    display(Image(filename=img_name))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)

# classifying image

    result = classifier.predict(test_image)
    training_set.class_indices
# Giving Labels

    if result[0][0] == 1:
        prediction = 'Pikachu'
    else:
        prediction = 'Iron Man '
    print(prediction)




# Getting all image file names from the test folder

import os
path = 'C:/Users/Hp/Desktop/iron/Dataset/test/test'
files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
   for file in f:
     if '.jpg' in file:
       files.append(os.path.join(r, file))
# Predicting and classifying each test image

for f in files:
   who(f)
   print('\n')





















