from flask import Flask, render_template,request
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import adam_v2
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator


app = Flask(__name__)
#specify the path to the train/test folders
train_dir = 'data/train/'
val_dir = 'data/test/'
#set image pixels to value of 1 or 0
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range = 10,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    fill_mode = 'nearest')

val_datagen = ImageDataGenerator(rescale=1./255)

#set image size/color/class for training and validation
train_generator = train_datagen.flow_from_directory(
        train_dir,
        #images in FER-2013 dataset are grayscale and 48x48
        target_size=(48,48),
        color_mode="grayscale",
        class_mode='binary')

validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48,48),
        color_mode="grayscale",
        class_mode='binary')


model = Sequential()

model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Dropout(0.3))

model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=adam_v2.Adam(learning_rate=0.0001),metrics=['accuracy'])
model_info = model.fit(
        train_generator,
        steps_per_epoch=12045// 64,
        epochs=100,
        validation_data=validation_generator,
        validation_steps= 3021 // 64)

model.save('kargocharlie.h5')
import cv2
@app.route('/')
def index():
   return render_template('home.html')



@app.route('/pred', methods=['GET', 'POST'])

def pred():
        if request.method == 'POST':
                # store image in static folder
                f = request.files['process_image']
                f.save('./static/process_image.jpg')
                # read image from static folder
                img = cv2.imread('./static/process_image.jpg')
                # resize image to 48x48
                img = cv2.resize(img, (48, 48))
                # convert image to grayscale
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # img = image.load_img("{{url_for('static',filename = 'images/download.jpg')}}",target_size = (48,48),color_mode = "grayscale")
                img = np.array(img)
                label_dict = {0:'happy',1:'sad'}

                img = np.expand_dims(img,axis = 0) #makes image shape (1,48,48)
                img = img.reshape(1,48,48,1)
                result = model.predict(img)
                result = list(result[0])
               
                if result[0] == 1:
                        print("sad")
                        return render_template('sad.html')
                else:
                        print("happy")
                        return render_template('happy.html')

if __name__ == '__main__':
   app.debug = True
   app.run()