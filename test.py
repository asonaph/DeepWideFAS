import math, re, os
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from utils import lbp_histogram
from utils import lbp 
print("Tensorflow version " + tf.__version__)


from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

img_augmentation = Sequential(
    [
        preprocessing.RandomFlip(mode='horizontal', seed=1996), 
    ],
    name="img_augmentation",
)

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
import tensorflow as tf
NUM_CLASSES = 2
IMG_SIZE = 256

def create_model():
  inputs_image = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
  # inputs_image = img_augmentation(inputs_image)
  resnet_model = ResNet50(include_top=False, input_tensor=inputs_image, weights="imagenet")
  resnet_model.trainable = True
  x = layers.GlobalAveragePooling2D(name="avg_pool")(resnet_model.output)
  x = layers.BatchNormalization()(x)
  outputs = layers.Dense(512)(x)
  model_image = tf.keras.Model(inputs_image, outputs, name="model_image")

  input_lbp = layers.Input(shape=(256))
  lbp_model = Sequential(
      [
      layers.Dense(512, activation="relu"), 
      layers.BatchNormalization(),
      layers.Dense(256), 
      ]
  )
  lbp_model(input_lbp)

  combined = layers.concatenate([model_image.output, lbp_model.output])
  # apply a FC layer and then a regression prediction on the
  # combined outputs
  z = layers.Dense(1024, activation="relu")(combined)
  z = layers.Dense(512, activation="relu")(z)
  top_dropout_rate = 0.05
  z = layers.Dropout(top_dropout_rate, name="top_dropout")(z)
  z = layers.Dense(NUM_CLASSES, activation="softmax", name="pred")(z)

  # our model will accept the inputs of the two branches and
  # then output a single value
  model = tf.keras.Model(inputs=[model_image.input, lbp_model.input], outputs=z)

  optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
  model.compile(
          optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=['accuracy']
      )
  
  return model

from mtcnn import MTCNN
from PIL import Image
import cv2
detector = MTCNN()


def get_face_pil(path):
  img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
  res = detector.detect_faces(img)
  x,y, width, height = res[0]['box']
  left = x; top = y; right = x + width; bottom = y + height
  face = Image.open(path).crop([left, top, right, bottom])
  return face

def get_lbp(face_pil):
  rgb_array = np.asarray(face_pil)
  return lbp_histogram(rgb_array, 8, 1)

def get_input(path):
  face = get_face_pil(path).resize((256,256))
  lbp = get_lbp(face)
  face_arr = np.expand_dims(np.asarray(face), axis=0)
  lbp_arr = np.expand_dims(lbp, axis=0)
  return face_arr, lbp_arr


face_arr, lbp_arr = get_input(sample_path)

live_score, spoof_score = model.predict([face_arr, lbp_arr])