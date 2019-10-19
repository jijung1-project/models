from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers
from keras.callbacks import LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
import numpy as np
from keras.utils import plot_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from google.colab import drive
import os
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras

reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, cooldown= 3,
                              patience=5, min_lr=0.0001)

def create_cnn(input_shape):
    cnn = Sequential()
    cnn.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001),
                   input_shape=input_shape))
    cnn.add(BatchNormalization())
    cnn.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)))
    cnn.add(BatchNormalization())
    cnn.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)))
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Dropout(0.2))

    cnn.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)))
    cnn.add(BatchNormalization())
    cnn.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)))
    cnn.add(BatchNormalization())
    cnn.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)))
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Dropout(0.3))

    cnn.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)))
    cnn.add(BatchNormalization())
    cnn.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)))
    cnn.add(BatchNormalization())
    cnn.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)))
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Dropout(0.4))



    cnn.add(Flatten())
    cnn.add(Dense(1024, activation='relu'))
    cnn.add(BatchNormalization())
    cnn.add(Dropout(0.5))
    cnn.add(Dense(150, activation='softmax'))
    # model summary
    cnn.summary()

    return cnn


def data_augmenation(X_train):
    # Data Augmentation
    augmentation = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
    )
    augmentation.fit(X_train)

    return augmentation


def main():
    drive.mount('/content/drive')
    X_train = np.load("/content/drive/My Drive/X_300_train.npy")
    X_test = np.load("/content/drive/My Drive/X_300_test.npy")
    y_train = np.load("/content/drive/My Drive/y_300_train.npy")
    y_test = np.load("/content/drive/My Drive/y_300_test.npy")
    print("OK")
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    print("OK")

    MODEL_SAVE_FOLDER_PATH = '/content/drive/My Drive/models/'
    if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
        os.mkdir(MODEL_SAVE_FOLDER_PATH)

    model_path = MODEL_SAVE_FOLDER_PATH + '{epoch:02d}-{val_acc:.4f}.hdf5'
    cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_acc', verbose=1, save_best_only=True)

    # early stop
    cb_early_stopping = EarlyStopping(monitor='val_acc', patience=20)

    cnn = create_cnn(X_train.shape[1:])
    plot_model(cnn, to_file='./model.png', show_shapes=True)
    # adam optimizer
    optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None)

    cnn.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    augmented_images = data_augmenation(X_train)
    hist = cnn.fit_generator(augmented_images.flow(X_train, y_train, batch_size=200),
                             steps_per_epoch=X_train.shape[0] / 200, epochs=200, verbose=1,
                             validation_data=(X_test, y_test), shuffle=True,
                             callbacks=[reduce_lr, cb_early_stopping,
                                        cb_checkpoint])
    # hist = cnn.fit(X_train, y_train, batch_size=50, epochs=200, verbose=1, validation_data=(X_test, y_test), callbacks=[reduce_lr, cb_early_stopping,cb_checkpoint] )
    test_accuracy = cnn.evaluate(X_test, y_test, verbose=1)
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