import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

BATCH_SIZE = 16
IMAGE_SIZE = (256, 256)

AUTOTUNE = tf.data.AUTOTUNE

def get_train_val_dataset():
	# Get train dataset
	train_ds = tf.keras.utils.image_dataset_from_directory(
		directory = "Defungi Split/Train",
		batch_size = BATCH_SIZE,
		image_size = IMAGE_SIZE,
		seed=27
		)

	# Get validation dataset
	val_ds = tf.keras.utils.image_dataset_from_directory(
		directory = "Defungi Split/Validation",
		batch_size = BATCH_SIZE,
		image_size = IMAGE_SIZE,
		seed=27
		)
	
	return train_ds, val_ds

def get_test_dataset():
	# Get train dataset
	test_ds = tf.keras.utils.image_dataset_from_directory(
		directory = "Defungi Split/Test",
		batch_size = BATCH_SIZE,
		image_size = IMAGE_SIZE,
		seed=27
		)

	return test_ds

def train_model(model, train_dataset, validation_dataset, epochs = 50, initial_epoch = 0, epochs_per_checkpoint = 5,
		load_checkpoint: str = None, save_checkpoint_folder = "checkpoints"):
	
	# Get dataset info
	batch_count = len(train_ds)

	# Load checkpoint if any
	if load_checkpoint:
		model.load_weights(load_checkpoint)
	
	# Create checkpoint callback
	save_checkpoint_path = save_checkpoint_folder + "/cp-{epoch:04d}.ckpt"
	model1_cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=save_checkpoint_path, save_weights_only=True, save_freq=epochs_per_checkpoint*batch_count, verbose=1)

	# Train the model
	model.fit(
		train_dataset,
		validation_data=validation_dataset,
		initial_epoch=initial_epoch,
		epochs=epochs,
		callbacks=[model1_cp_callback],
		use_multiprocessing=True
	)

def get_base_model(num_classes, checkpoint_path: str = None):
	# Import pre-trained model
	os.environ['TFHUB_CACHE_DIR'] = 'tf_cache'
	feature_extractor_model = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b2/feature_vector/2"
	feature_extractor_layer = hub.KerasLayer(
		feature_extractor_model,
		input_shape=IMAGE_SIZE+(3,),
		trainable=False)
	
	# Create model
	model = Sequential([
		layers.Rescaling(1./255, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
		feature_extractor_layer,
		layers.Flatten(),
		layers.Dense(num_classes)])
	model.compile(optimizer='adam',
				loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
				metrics=['accuracy'])
	model.summary()

	# Load checkpoint if any
	if checkpoint_path:
		model.load_weights(checkpoint_path)

	return model

if __name__ == "__main__":
	train_ds, val_ds = get_train_val_dataset()
	
	class_names = train_ds.class_names
	num_classes = len(class_names)

	# Configure datasets for optimization
	train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
	val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

	# Create base model
	base_model = get_base_model(num_classes, "checkpoints/base-model/cp-0050.ckpt")

	# Load from checkpoint
	# train_model(base_model, train_ds, val_ds, 50, 0, 5, "LOAD_CHECKPOINT", "checkpoints/base-model")

	test_ds = get_test_dataset()
	test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)


	base_model.evaluate(test_ds)

	

	
	

	

	

	