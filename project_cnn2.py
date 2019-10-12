import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar100
from keras import regularizers
from keras.callbacks import LearningRateScheduler, EarlyStopping
import numpy as np
from keras.utils import plot_model
import matplotlib.pyplot as plt
import glob
import os.path as path
from scipy import misc
import os
import glob
from skimage.transform import resize
import skimage.io as io
from skimage.io import imsave
from PIL import Image
import imageio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from google.colab import drive


# def imageLoad(dirname):
#     images = list()
#     filenames = os.listdir(dirname)
#     y = list()
#
#     for filename in filenames:
#         if filename != '.DS_Store':
#             full_dirname = os.path.join(dirname, filename)
#             filenames = os.listdir(full_dirname)
#             for i in range(0, 300):
#                 img = imageio.imread(full_dirname + '/' + filenames[i])
#                 images.append(img)
#             for j in range(0, 300):
#                 y.append(filename)
#     images = np.asarray(images)
#     y = np.asarray(y)
#     image_size = np.asarray([images.shape[1], images.shape[2], images.shape[3]])
#
#     le = LabelEncoder()
#     le.fit(y)
#     y = le.transform(y)
#     return images, y


def learning_rate_scheduler(epoch_counter):
    learning_rate = 0.001
    if epoch_counter > 75: learning_rate = 0.0005
    if epoch_counter > 100: learning_rate = 0.0003
    return learning_rate


def prepare_data():
    # X, y = imageLoad("/Users/jaewan/Desktop/food_img")
    # np.save("/Users/jaewan/Desktop/X", X)
    # np.save("/Users/jaewan/Desktop/y", y)
    X = np.load('/content/drive/My Drive/X.npy')
    y = np.load('/content/drive/My Drive/y.npy')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    return (X_train, y_train), (X_test, y_test)


def create_cnn(input_shape):
    cnn = Sequential()
    cnn.add(Conv2D(32, (3, 3), activation='elu', padding='same', kernel_regularizer=regularizers.l2(0.0001),
                   input_shape=input_shape))
    cnn.add(BatchNormalization())
    cnn.add(Conv2D(32, (3, 3), activation='elu', padding='same', kernel_regularizer=regularizers.l2(0.0001)))
    cnn.add(BatchNormalization())
    cnn.add(Conv2D(32, (3, 3), activation='elu', padding='same', kernel_regularizer=regularizers.l2(0.0001)))
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Dropout(0.2))

    cnn.add(Conv2D(64, (3, 3), activation='elu', padding='same', kernel_regularizer=regularizers.l2(0.0001)))
    cnn.add(BatchNormalization())
    cnn.add(Conv2D(64, (3, 3), activation='elu', padding='same', kernel_regularizer=regularizers.l2(0.0001)))
    cnn.add(BatchNormalization())
    cnn.add(Conv2D(64, (3, 3), activation='elu', padding='same', kernel_regularizer=regularizers.l2(0.0001)))
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Dropout(0.3))

    cnn.add(Conv2D(128, (3, 3), activation='elu', padding='same', kernel_regularizer=regularizers.l2(0.0001)))
    cnn.add(BatchNormalization())
    cnn.add(Conv2D(128, (3, 3), activation='elu', padding='same', kernel_regularizer=regularizers.l2(0.0001)))
    cnn.add(BatchNormalization())
    cnn.add(Conv2D(128, (3, 3), activation='elu', padding='same', kernel_regularizer=regularizers.l2(0.0001)))
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Dropout(0.4))
    cnn.add(Flatten())

    cnn.add(Dense(30, activation='softmax'))
    # model summary
    cnn.summary()

    return cnn


def data_augmenation(X_train):
    # Data Augmentation
    augmentation = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )
    augmentation.fit(X_train)

    return augmentation


def main():
    drive.mount('/content/drive')

    (X_train, Y_train), (X_test, Y_test) = prepare_data()

    # early stop
    cb_early_stopping = EarlyStopping(monitor='val_loss', patience=20)

    cnn = create_cnn(X_train.shape[1:])
    plot_model(cnn, to_file='./model.png', show_shapes=True)
    # adam optimizer
    optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    cnn.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    augmented_images = data_augmenation(X_train)
    hist = cnn.fit_generator(augmented_images.flow(X_train, Y_train, batch_size=64),
                             steps_per_epoch=X_train.shape[0] / 64, epochs=125, verbose=1,
                             validation_data=(X_test, Y_test),
                             callbacks=[LearningRateScheduler(learning_rate_scheduler), cb_early_stopping])

    test_accuracy = cnn.evaluate(X_test, Y_test, verbose=1)
    print('test accuracy: %.3f | test loss: %.3f' % (test_accuracy[1] * 100, test_accuracy[0]))

    # print loss and accuracy in plot
    fig, loss_ax = plt.subplots()

    acc_ax = loss_ax.twinx()

    loss_ax.plot(hist.history['loss'], 'y', label='train loss')
    loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

    acc_ax.plot(hist.history['acc'], 'b', label=' acc')
    acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')

    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    acc_ax.set_ylabel('accuray')

    loss_ax.legend(loc='upper left')
    acc_ax.legend(loc='lower left')

    plt.show()


if __name__ == '__main__':
    main()
