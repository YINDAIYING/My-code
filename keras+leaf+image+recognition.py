
# coding: utf-8

# In[65]:

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


datagen = ImageDataGenerator(
        rotation_range=50,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.4,
        zoom_range=0.4,
        horizontal_flip=True,
        fill_mode='nearest')


# In[ ]:





# In[66]:

batch_size = 25

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.25,
        zoom_range=0.25,
        horizontal_flip=True
        )

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        "C:/Users/尹代英/Desktop/Convoluntional Neural Network/Convoluntional Neural Network/Training Images2",  # this is the target directory
        target_size=(100, 100),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='categorical')  


# In[ ]:




# In[69]:

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        'C:/Users/尹代英/Desktop/Convoluntional Neural Network/Convoluntional Neural Network/Validation Images2',
        target_size=(100, 100),
        batch_size=batch_size,
        class_mode='categorical')


# In[70]:

img = load_img('C:/Users/尹代英/Desktop/Convoluntional Neural Network/Convoluntional Neural Network/Training Images2\BabyBibs\BabyBibs_611.jpg')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)

x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='C:/Users/尹代英/Desktop/Convoluntional Neural Network/Convoluntional Neural Network/generated images',
                          save_prefix='fashion', save_format='jpeg'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely


# In[75]:

print(x.shape)


# In[81]:

model = Sequential()

model.add(Conv2D(50, (3, 3), input_shape=(100, 100, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))


model.add(Conv2D(50, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))


model.add(Conv2D(50, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# In[82]:

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))

model.add(Activation('relu'))


model.add(Dropout(0.5))
model.add(Dense(18))

model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


# In[85]:

bacth_size = 25

model.fit_generator(
        train_generator,
        steps_per_epoch=10000// batch_size,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=10000 // batch_size)
model.save('first_try2.h5')



