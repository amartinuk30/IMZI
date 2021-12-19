# # Speech Recognition

# Dataset Used:
# - A simple audio/speech dataset consisting of recordings of spoken digits in wav files at 8kHz.
# - 2,000 recordings (50 of each digit per speaker)

# Importing all the required libraries
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings('ignore')

### Distribution of each label of sound

path = r'C:\Users\amart\Desktop\Speech_recognition_MLP\Data'
labels = os.listdir(path)
# find count of each label and plot bar graph
labels = []
number_of_records = []
for f in os.listdir(path):
    if f.endswith('.wav'):
        label = f.split('_')[0]
        labels.append(label)

label_dic = {}
for i in labels:
    if i in label_dic:
        label_dic[i] += 1
    else:
        label_dic[i] = 1

plt.bar(label_dic.keys(), label_dic.values())
plt.title('Number of recordings per digit')
plt.xlabel('Digit', fontsize=12)
plt.ylabel('Number of audio recordings', fontsize=12)
plt.show()


#### Data Preprocessing
# - I need to vectorize the sound samples in order to put them in the model.

####  Creating a function to get the MFCCs of an audio signal
def get_mfcc(file_path, max_pad_len=20):
    x, sr = librosa.load(file_path, mono=True, sr=None)

    # Jumping every 3rd index of the array
    x = x[::3]

    # Getting MFCCs for the audio signal
    mfcc = librosa.feature.mfcc(x, sr=8000)

    # Calculating the padding width to make number of rows equal to number of column in the feature matrix
    pad_width = max_pad_len - mfcc.shape[1]

    # Making mfcc a square-matrix
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfcc


# ####  Creating a function to get the MFCCs of all the audio signals and converting them into an array, also extracting labels out of each audio file and storing it in labels
# - MFCC of each audio file is of shape 20x20
# - MFCCs of all the audio files after converting into array will be of shape (2000,20,20)
# - Similarly converting 2000 labels with 10 classes into an array will result in an array of shape 2000x10

def get_data():
    labels = []
    mfccs = []
    for i in os.listdir('./Data'):
        if i.endswith('.wav'):
            # Extracting the MFCC features of each audio file and storing it
            mfccs.append(get_mfcc('./Data/' + i))
            # Extracting the label of each audio file and appending it
            label = i.split('_')[0]
            labels.append(label)
    return np.asarray(mfccs), to_categorical(labels)


### Modeling
# - Designing a Multilayer Perceptron using Convolution Neural Network

def cnn_mlp_model(input_shape, num_classes):
    # Defining a sequential model
    model = Sequential()

    # Feature Learning Layers  -- Convolution - RELU 1st Block, Minimum of three such blocks is what I go with a rule of thumb

    # This is the input layer, so input shape has to be given, this layer uses 32 feature maps of size 2x2
    model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())

    # Convolution - RELU 2nd Block. This layer uses 64 feature maps of size 3x3
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())

    # Convolution - RELU 3rd Block, using 128 feature maps, size 5x5
    model.add(Conv2D(128, kernel_size=(5, 5), activation='relu'))
    model.add(BatchNormalization())

    # I tried using same kernel size with different number of feature maps but this is the combination with which I got the
    # best results in limited amount of time

    # Max Pooling layer with pooling matrix of size 2x2
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Flattening the vector to input it into the model
    model.add(Flatten())

    # Classification layers

    # First hidden layer has 128 Neurons and it also represents the dimension of its output space
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    # Second hidden layer has 64 Neurons and it is the dimension of its output space.
    # Most of the problems are solved using 1 to 2 hidden layers
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    # This is the final layer in the model, it has num_classes number of neurons which is 10 in our case. Activation function
    # used here is softmax as here it will classify the input into on of the 10 output classes
    model.add(Dense(num_classes, activation='softmax'))

    # Loss function used is categorical_crossentropy as it is a multiclass-classification problem
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])

    return model


# #### Creating a function that extracts all the labels and MFCC features from the audio dataset,
# splits the data into training and test set and complies the Neural Network we created above
def get_all():
    # Getting the mfcc feature set and labels from the dataset
    mfccs, labels = get_data()
    dimension_1 = mfccs.shape[1]
    dimension_2 = mfccs.shape[2]
    # Total categories of the labels (0-9)
    classes = 10
    X = mfccs
    X = X.reshape((mfccs.shape[0], dimension_1, dimension_2, 1))
    y = labels

    input_shape = (dimension_1, dimension_1, 1)

    # Splitting the data into train and test with 20% of the data as test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # Compiling the Model
    model = cnn_mlp_model(input_shape, classes)

    return X_train, X_test, y_train, y_test, model


### Model Training

# Calling the get_all function to get the training and test set, and the compiled Neural Network model
X_train, X_test, y_train, y_test, cnn_mlp_model = get_all()

# We are monitoring validation loss and creating a check point, so that only when there is an improvement in the validation
# accuracy, the model is saved.
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=7)
mc = ModelCheckpoint(r'C:\Users\amart\Desktop\Speech_recognition_MLP\model.h5', monitor='val_accuracy', verbose=1,
                     save_best_only=True, mode='max')

history = cnn_mlp_model.fit(X_train, y_train, epochs=20, callbacks=[es, mc], batch_size=64, verbose=1, validation_split=0.2)

# ### Plotting the training loss and validation loss with number of epochs
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.legend()
plt.show()


