#!/usr/bin/env python
# coding: utf-8

# Experiment 2 used the entire uncropped swing event images to perform final classification
# the following models were used:
# * LinearSVM
# * CatBoost
# Mini-VGG

# In[1]:
import os

# remove annoying tensorflow warnings (must be before tensorflow or keras imports)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import Input
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical
from sklearn.utils import Bunch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn import metrics
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
from keras.optimizers.optimizer_v2.adam import Adam
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import LinearSVC
from catboost import CatBoostClassifier

print("Num GPUs Available", len(tf.config.list_physical_devices('GPU')))
print(device_lib.list_local_devices())

rseed = 42

flat_data = 'input/flat_dat.npy'
target = 'input/target.npy'
target_names = 'input/target_names.npy'
images = 'input/images.npy'
DESCR = 'input/descr.npy'

# load numpy arrays of golf swing image data and labels
swing_image_dataset = Bunch()
swing_image_dataset['data'] = np.load(flat_data)
swing_image_dataset['target'] = np.load(target)
swing_image_dataset['target_names'] = np.load(target_names)
swing_image_dataset['images'] = np.load(images)
swing_image_dataset['DESCR'] = np.load(DESCR)

print("model loaded")
# In[4]:


X = MinMaxScaler().fit_transform(swing_image_dataset.data)

# ### Dimensionality Reduction 
# #### LDA using 7 components
# In[6]:

plt.figure(figsize=(20, 15))
for i, target_name in zip([0, 1, 2, 3, 4, 5, 6, 7], swing_image_dataset.target_names):
    plt.scatter(X[swing_image_dataset.target == i, 0], X[swing_image_dataset.target == i, 1], label=target_name)
plt.legend(fontsize=25)
plt.title('LDA of GolfDB dataset')

plt.show()

# 80:20 train:test split
X_train, X_test, y_train, y_test = train_test_split(X, swing_image_dataset.target, test_size=0.2, random_state=rseed)

lda = LDA(n_components=7)
X_train = lda.fit(X_train, y_train).transform(X_train)
X_test = lda.transform(X_test)

print("feature extraction done")
# In[9]:
accuracy = []
f1_score = []
precision_score = []
recall_score = []

# ### Classification Models
# #### LinearSVM

# In[10]:
print("LinearSVM:")

linearsvm = LinearSVC(random_state=rseed, C=10, multi_class='crammer_singer')
linearsvm.fit(X_train, y_train)
y_pred = linearsvm.predict(X_test)

print(metrics.classification_report(y_test, y_pred, target_names=swing_image_dataset.target_names))
accuracy_linearsvm = metrics.accuracy_score(y_test, y_pred)
f1_linearsvm = metrics.f1_score(y_test, y_pred, average='weighted')
precision_linearsvm = metrics.precision_score(y_test, y_pred, average='weighted')
recall_linearsvm = metrics.recall_score(y_test, y_pred, average='weighted')
print("Accuracy", accuracy_linearsvm * 100)
accuracy.append(accuracy_linearsvm * 100)
f1_score.append(f1_linearsvm * 100)
precision_score.append(precision_linearsvm * 100)
recall_score.append(recall_linearsvm * 100)

# # In[11]:
#
#
# from sklearn.metrics import ConfusionMatrixDisplay
# cm = metrics.confusion_matrix(y_test, y_pred)
# cm_display = ConfusionMatrixDisplay(cm, display_labels=swing_image_dataset.target_names)
# cm_display.plot(xticks_rotation = 'vertical', cmap='binary')
#

#
# # #### CatBoost
#
# # In[12]:
#
print("CatBoost:")

cat_clf = CatBoostClassifier(random_state=rseed,
                             depth=4,
                             bagging_temperature=1.5,
                             learning_rate=0.03,
                             l2_leaf_reg=7,
                             eval_metric='Accuracy',
                             task_type="GPU",
                             use_best_model=True,
                             verbose=False)

cat_clf.fit(X_train, y_train, eval_set=(X_test, y_test))

y_pred_cat = cat_clf.predict(X_test)
print(metrics.classification_report(y_test, y_pred_cat, target_names=swing_image_dataset.target_names))
accuracy_cat = metrics.accuracy_score(y_test, y_pred_cat)
f1_cat = metrics.f1_score(y_test, y_pred_cat, average='weighted')
precision_cat = metrics.precision_score(y_test, y_pred_cat, average='weighted')
recall_cat = metrics.recall_score(y_test, y_pred_cat, average='weighted')
print("Accuracy", accuracy_cat * 100)
accuracy.append(accuracy_cat * 100)
f1_score.append(f1_cat * 100)
precision_score.append(precision_cat * 100)
recall_score.append(recall_cat * 100)

print("CNN:")

# mini vgg consisting of first 2 blocks of the VGG neural network and batch normalization
def create_mini_vgg(input_shape=(32, 32, 1), cnum=10, dropout_rate=0.25,
                    neurons=32, include_top=True, weights='imagenet'):
    model = Sequential()

    # first CONV => RELU => CONV => RELU => POOL layer set
    model.add(Conv2D(neurons, (3, 3), padding="same", input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(Conv2D(neurons, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))

    # second CONV => RELU => CONV => RELU => POOL layer set
    model.add(Conv2D(neurons * 2, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(Conv2D(neurons * 2, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))

    # first (and only) set of FC => RELU layers, second fc2 doesn't help
    model.add(Flatten())
    model.add(Dense(neurons ** 2))
    model.add(Activation("relu"))  # added?
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    # softmax classifier
    model.add(Dense(cnum))
    model.add(Activation("softmax"))

    # return the constructed network architecture
    return model


X_keras = swing_image_dataset.images
cnum = len(np.unique(swing_image_dataset.target))
print(cnum)
le = LabelEncoder()
labels = le.fit_transform(swing_image_dataset.target)
labels = to_categorical(labels)
INPUT_SHAPE = [30, 30, 3]
image_input = Input(shape=(30, 30, 3))
EPOCHS = 200
LR = 0.0001  # as low as possible, local minima and time permitting
BS = 8  # 8 gives great results for these kinds of datasets but is super slow. larger the batch the fastest but more mem
X_train, X_test, y_train, y_test = train_test_split(np.float32(X_keras), labels, test_size=0.2, random_state=rseed)
print(X_train.shape)

X_train = X_train.reshape(X_train.shape[0], *INPUT_SHAPE)
X_test = X_test.reshape(X_test.shape[0], *INPUT_SHAPE)

# load model without output layer (because golfdb dataset has different labels to imagenet)
# also input_tensor so that you can specify the resolution and channels (stick to colour)


model = create_mini_vgg(input_shape=INPUT_SHAPE, cnum=y_train.shape[1], dropout_rate=0.1, neurons=32)
model.compile(optimizer=Adam(learning_rate=LR, decay=LR / EPOCHS), loss="categorical_crossentropy",
              metrics=["accuracy"])
# earlystopping to prevent overtraining
filepath = "checkpoints/"
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath, save_weights_only=True, save_best_only=True, verbose=2, monitor ="val_loss"),
    EarlyStopping(monitor='val_accuracy', mode='max', patience=10, restore_best_weights=True, verbose=1),
    keras.callbacks.TensorBoard( "logs_training/")
]
# With data augmentation to prevent overfitting
gen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    # rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range=0.05,  # Randomly zoom image
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False)  # randomly flip images

train_gen = gen.flow(X_train, y_train, batch_size=BS)
# without augmentation (achieves same as SVM and CATB):
# model.fit(X_train, y_train, batch_size=BS, epochs=EPOCHS, verbose=2, validation_data=(X_test, y_test), callbacks=callbacks)
# with augmentation
model.fit(train_gen, batch_size=BS, epochs=EPOCHS, verbose=2, validation_data=(X_test, y_test), callbacks=callbacks)

y_pred = model.predict(X_test)
y_test, y_pred = np.argmax(y_test, axis=-1), np.argmax(y_pred, axis=1)  # magical fix

print(metrics.classification_report(y_test, y_pred, target_names=swing_image_dataset.target_names))
accuracy_cnn = metrics.accuracy_score(y_test, y_pred)
f1_cnn = metrics.f1_score(y_test, y_pred, average='weighted')
precision_cnn = metrics.precision_score(y_test, y_pred, average='weighted')
recall_cnn = metrics.recall_score(y_test, y_pred, average='weighted')
print("Accuracy", accuracy_cnn * 100)
accuracy.append(accuracy_cnn * 100)
f1_score.append(f1_cnn * 100)
precision_score.append(precision_cnn * 100)
recall_score.append(recall_cnn * 100)

print(accuracy)
print(f1_score)
print(precision_score)
print(recall_score)

# ### Visualise Results

# In[18]:
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(20, 10))
data = [accuracy, f1_score]
key = ['Accuracy', 'F1-Score']
labels = ['LinearSVM', 'CatBoost', 'CNN']
bp_dict = plt.bar(labels, list(map(float, data[0])), align='edge', width=-0.2, color=['green'])
bp_dict = plt.bar(labels, list(map(float, data[1])), align='edge', width=0.2, color=['blue'])
plt.ylabel("Score(%)", fontsize=30)
plt.xlabel("Classification Model", fontsize=30)
plt.legend(key, fontsize=26)
plt.title("Experiment 2: Accuracy and F1 Scores", fontsize=34)
plt.ylim((0, 100))
plt.tick_params(labelsize=26)
plt.grid()
plt.show()
plt.savefig("results.jpg")
