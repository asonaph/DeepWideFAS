from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers


NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)
NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
print('Dataset: {} training images, {} unlabeled test images'.format(NUM_TRAINING_IMAGES, NUM_TEST_IMAGES))

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


with strategy.scope():
  model = create_model()


checkpoint_filepath = "gs://result_fasd_lbp81"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    save_best_only=True)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
history = model.fit(
    get_training_dataset(), 
    validation_data = get_validation_dataset(),
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=60, 
    callbacks=[early_stop, model_checkpoint_callback]
)