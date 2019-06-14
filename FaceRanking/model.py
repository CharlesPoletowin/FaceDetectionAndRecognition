import pandas as pd
import os
from PIL import Image
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

import keras
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Dropout,Flatten
from keras.optimizers import Adam
vgg=VGG16(weights='imagenet', pooling='max')


def show_some(features, flag):
    for x in range(0, 5):
        single_fea = features[int(x * 100)]
        img = Image.fromarray(single_fea)
        if flag == 1:
            img.save(str(x) + '.jpg')
        if flag == 0:
            img.save(str(x) + '_raw.jpg')


def read_image1(dir_list):
    feature_list = []
    for image_dir in dir_list:
        img = Image.open(image_dir)
        img = img.resize((224, 224))
        im = np.asarray(img)
        feature_list.append(im)
    raw_features = np.array(feature_list)

    return raw_features

def read_image(image_directory='data/'):
    # features=np.zeros((300,400))
    feature_list = []
    image_directory=Path(image_directory)
    for image_path in image_directory.glob("*.jpg"):
        img = Image.open(image_path)
        img = img.resize((224, 224))
        im = np.asarray(img)
        feature_list.append(im)
    raw_features = np.array(feature_list)
    return raw_features

    #
    # adjust these codes in order to deal with the large amount of photos in the same directory
    # for image_dir in dir_list:
    #     img = Image.open(image_dir)
    #     img = img.resize((224, 224))
    #     im = np.asarray(img)
    #     # print(im.shape)
    #     # print(img.size)
    #     # new_img = Image.fromarray(im)
    #     # new_img.show()
    #
    #     # features=np.stack(features,im)
    #     feature_list.append(im)
    # raw_features = np.array(feature_list)
    # # normalized_features=raw_features.astype('float32')/255.0
    # return raw_features


def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('history')
    plt.ylabel(train)
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()

data_dir='data/'
df = pd.read_excel('Attractiveness label.xlsx')

ndarray = df.to_numpy()

label = ndarray[:, 1]
label = np.reshape(label, (500, 1))

image_flag = ndarray[:, 0]
dir_list = []

for flag in image_flag:
    # print(flag)
    flag_convert = str(int(flag))
    image_dir = data_dir + 'SCUT-FBP-' + flag_convert + '.jpg'
    dir_list.append(image_dir)

normalized_features = read_image1(dir_list)
train_data=normalized_features[:400]
train_label=label[:400]

#Img=Image.fromarray(train_data[3].astype('uint8')*255)
#Img.save('WTF.png')
test_data=normalized_features[400:]
test_label=label[400:]

model = Sequential()
model.add(vgg)
model.add(Dense(1))
model.layers[0].trainable = False

# build CNN

model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001),metrics=['mae'])
earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
history=model.fit(batch_size=20, x=train_data, y=train_label, callbacks=[earlyStopping], validation_split=0.1,epochs=1000,verbose=2)
#model.fit(train_data, train_label, batch_size=28, nb_epoch=500, verbose=2,  validation_split=0.1)
scores=model.evaluate(x=test_data,y=test_label)
print (scores[1])
#show_train_history(history,'mae','val_mae')

prediction=model.predict(test_data)
image_directory='test/'
test =read_image(image_directory)

predict=model.predict(test)
print(predict)
model.save('face_model.h5')