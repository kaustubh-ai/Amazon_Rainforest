import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import keras
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras.models import Sequential
from sklearn.metrics import fbeta_score
from keras_preprocessing.MLB import ImageDataGenerator
from sklearn.preprocessing import MultiLabelBinarizer
from keras.applications.vgg16 import VGG16, preprocess_input
# from keras.applications.xception import Xception, preprocess_input
from skmultilearn.model_selection import iterative_train_test_split
from keras.preprocessing.image import ImageDataGenerator as ImageDataGen
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau
from keras.layers import Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D, \
	Input

xx, yy, channel, epochs = 156, 156, 'rgb', 50
data_dir = r'C:\Users\Kaustubh\zzz\resources\Planets'


def df_create(x, y, mlb):
	return pd.DataFrame({'Image': x.reshape(x.shape[0]),
						 'Label': mlb.inverse_transform(y)})


def multi_split(df_, img_format='png'):
	col_0, col_1 = df_.columns[0], df_.columns[1]

	df_[col_1] = df_[col_1].str.split(' ')
	df_[col_0] = df_[col_0].apply(lambda x: x + '.{}'.format(img_format))

	x = df_[col_0].values
	x = x.reshape(x.shape[0], 1)
	mlb = MultiLabelBinarizer()
	y = mlb.fit_transform([i for i in df[col_1]])

	x_train, y_train, x_test, y_test = iterative_train_test_split(x, y, test_size=0.94)

	df_train_ = df_create(x_train, y_train, mlb)
	df_test_ = df_create(x_test, y_test, mlb)

	return df_train_, df_test_


df = pd.read_csv(os.path.join(data_dir, 'train_v2.csv'))
df_train, df_val = multi_split(df, img_format='jpg')

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_iter = train_datagen.flow_from_directory(os.path.join(data_dir, 'data'), target_size=(xx, yy), batch_size=56,
                                               shuffle=True, dataframe=df_train)
val_iter = val_datagen.flow_from_directory(os.path.join(data_dir, 'data'), target_size=(xx, yy), batch_size=5,
                                           shuffle=False, dataframe=df_val)
test_iter = test_datagen.flow_from_directory(os.path.join(data_dir, 'breed\\test'), target_size=(xx, yy), batch_size=35,
                                             shuffle=False, dataframe=df)

train_steps = train_iter.n // train_iter.batch_size
val_steps = val_iter.n // val_iter.batch_size
test_steps = test_iter.n // test_iter.batch_size
