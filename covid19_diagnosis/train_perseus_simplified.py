# mpirun -np 4 --allow-run-as-root python train_perseus_v2.py -e 50 -bs 8

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
import perseus.tensorflow.horovod.keras as hvd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import time
import os
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-lr", "--initial_lr", type=float, default=0.0001, help="initial learning rate (default 1e-3)")
ap.add_argument("-e", "--epochs", type=int, default=50, help="epoch size")
ap.add_argument("-bs", "--batch_size", type=int, default=8, help="batch size")
ap.add_argument("-s", "--step_size", type=int, required=False, help="step size")
args = vars(ap.parse_args())

# initialize Horovod
hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


def construct_model(initial_lr, num_epochs):
	# load Resnet50 network with pre-trained ImageNet, lay off head FC layer
	baseModel = ResNet50(weights="imagenet", 
					  include_top=False, 
					  input_tensor=Input(shape=(224, 224, 3)))
	
	# construct new head for fine-tuning
	headModel = baseModel.output
	headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
	headModel = Flatten(name="flatten")(headModel)
	headModel = Dense(256, activation="relu")(headModel)
	headModel = Dropout(0.5)(headModel)
	headModel = Dense(len(CLASSES), activation="softmax")(headModel)

	# place the head FC model on top of the base model
	model = Model(inputs=baseModel.input, outputs=headModel)

	# freeze base model layers
	for layer in baseModel.layers:
		layer.trainable = False

	# compile model: Horovod
	opt = Adam(lr=initial_lr * hvd.size(), decay=initial_lr / num_epochs)
	
	opt = hvd.DistributedOptimizer(opt)

	model.compile(loss="binary_crossentropy", 
				  optimizer=opt, 
				  metrics=["accuracy"],
				  experimental_run_tf_function=False)

	callbacks = [
		# Horovod: broadcast initial variable states from rank 0 to all other processes.
		hvd.callbacks.BroadcastGlobalVariablesCallback(0),
		# Horovod: average metrics among workers at the end of every epoch.
		hvd.callbacks.MetricAverageCallback()
	]

	return model, callbacks


def serialize_model(model, outputPath):
	print("[INFO] saving COVID-19 diagnostic model...")
	model.save(outputPath, save_format="h5")


# runtime params
BS = args['batch_size']
INIT_LR = args['initial_lr']
NUM_EPOCHS = args['epochs']
STEP_SIZE = args['step_size']
MODEL_PATH = 'covid19_resnet50.model'
PLOT_PATH = 'covid19_resnet50_plot.png'
CLASSES = ["covid", "normal"]

# load dataset
TRAIN_PATH = 'chest_images/training'
VAL_PATH = 'chest_images/validation'
TEST_PATH = 'chest_images/test'

print('[INFO] loading images...', end='')
imagePaths = list(paths.list_images(TRAIN_PATH))
images = []
labels = []

for imagePath in imagePaths:
	# extract the class label from the filename
	label = imagePath.split(os.path.sep)[-2]

	# load the image, swap color channels, and resize it to be a fixed
	# 224x224 pixels while ignoring aspect ratio
	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (224, 224))

	# update the data and labels lists, respectively
	images.append(image)
	labels.append(label)

print(f'Done! {len(images)}/{len(labels)} training images')

# convert to np arrays and normalize it to rgb range [0, 255]
data = np.array(images) / 255.0
labels = np.array(labels)

# one-hot encode labels by directory name
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# train/test split (90:10)
(trainX, testX, trainY, testY) = train_test_split(data, 
												  labels,
												  test_size=0.80, 
												  stratify=labels, 
												  random_state=42)


# construct model
print(f'[INFO] GPU {hvd.rank()} -> building model...')
model, callbacks = construct_model(INIT_LR, NUM_EPOCHS)

# fine-tune the network
print(f'[INFO] GPU {hvd.rank()} -> training model...')

t0 = time.time()
H = model.fit(
	trainX, trainY,
	batch_size = BS,
    epochs = NUM_EPOCHS,
	verbose = 1 if hvd.rank() == 0 else 0,
	callbacks = callbacks)
t1 = time.time()

if hvd.rank() == 0:
	print(f"[INFO] Completed {NUM_EPOCHS} epochs in {(t1-t0):.1f} sec using BATCH SIZE {BS}")