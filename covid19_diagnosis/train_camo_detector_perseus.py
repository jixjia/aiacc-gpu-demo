# USAGE
# python train_camo_detector.py

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
import perseus.tensorflow.horovod.keras as hvd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import time

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-lr", "--initial_lr", type=float, default=0.001, help="initial learning rate (default 1e-3)")
ap.add_argument("-e", "--epochs", type=int, default=50, help="epoch size")
ap.add_argument("-bs", "--batch_size", type=int, default=8, help="batch size")
ap.add_argument("-s", "--step_size", type=int, default=50, help="step size")
args = vars(ap.parse_args())

# initialize Horovod
hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

# determine the total number of image paths in training, validation,
# and testing directories
TRAIN_PATH = 'chest_images/training'
VAL_PATH = 'chest_images/validation'
TEST_PATH = 'chest_images/test'
BS = args['batch_size']
INIT_LR = args['initial_lr']
NUM_EPOCHS = args['epochs']
MODEL_PATH = 'covid19_resnet50.model'
CLASSES = ["covid", "normal"]

totalTrain = len(list(paths.list_images(TRAIN_PATH)))
totalVal = len(list(paths.list_images(VAL_PATH)))
totalTest = len(list(paths.list_images(TEST_PATH)))

print(f'[INFO] GPU {hvd.rank()} -> \nTrainset: {totalTrain}\nValset: {totalVal}\nTestset: {totalTest}')

# initialize the training training data augmentation object
trainAug = ImageDataGenerator(
	rotation_range=25,
	zoom_range=0.1,
	width_shift_range=0.1,
	height_shift_range=0.1,
	shear_range=0.2,
	horizontal_flip=True,
	fill_mode="nearest")

# initialize the validation/testing data augmentation object (which
# we'll be adding mean subtraction to)
valAug = ImageDataGenerator()

# define the ImageNet mean subtraction (in RGB order) and set the
# the mean subtraction value for each of the data augmentation
# objects
mean = np.array([123.68, 116.779, 103.939], dtype="float32")
trainAug.mean = mean
valAug.mean = mean

# initialize the training generator
trainGen = trainAug.flow_from_directory(
	TRAIN_PATH,
	class_mode="categorical",
	target_size=(224, 224),
	color_mode="rgb",
	shuffle=True,
	batch_size=BS)

# initialize the validation generator
valGen = valAug.flow_from_directory(
	VAL_PATH,
	class_mode="categorical",
	target_size=(224, 224),
	color_mode="rgb",
	shuffle=False,
	batch_size=BS)

# initialize the testing generator
testGen = valAug.flow_from_directory(
	TEST_PATH,
	class_mode="categorical",
	target_size=(224, 224),
	color_mode="rgb",
	shuffle=False,
	batch_size=BS)

# load the ResNet-50 network, ensuring the head FC layer sets are left off
print(f'[INFO] GPU {hvd.rank()} -> preparing model...')
baseModel = ResNet50(weights="imagenet", include_top=False,	input_tensor=Input(shape=(224, 224, 3)))

# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(256, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(len(CLASSES), activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the training process
for layer in baseModel.layers:
	layer.trainable = False

# compile the model
opt = Adam(lr=INIT_LR * hvd.size(), decay=INIT_LR / NUM_EPOCHS)
opt = hvd.DistributedOptimizer(opt)

model.compile(
	loss="binary_crossentropy", 
	optimizer=opt,
	metrics=["accuracy"],
	experimental_run_tf_function=False)

callbacks = [
		# Horovod: broadcast initial variable states from rank 0 to all other processes.
		hvd.callbacks.BroadcastGlobalVariablesCallback(0),
		# Horovod: average metrics among workers at the end of every epoch.
		hvd.callbacks.MetricAverageCallback(),
		# hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=3, verbose=1),
	]

# train the model
print(f'[INFO] GPU {hvd.rank()} -> training model...')
t0 = time.time()
H = model.fit_generator(
	trainGen,
	steps_per_epoch=totalTrain // BS  // hvd.size(),
	validation_data=valGen,
	validation_steps=totalVal // BS  // hvd.size(),
	epochs=NUM_EPOCHS,
	verbose = 1 if hvd.rank() == 0 else 0,
	callbacks = callbacks)
t1 = time.time()

if hvd.rank() == 0:
	# reset the testing generator and then use our trained model to
	# make predictions on the data
	print("[INFO] evaluating network...")
	testGen.reset()
	predIdxs = model.predict_generator(testGen, steps=(totalTest // BS) + 1)

	# for each image in the testing set we need to find the index of the
	# label with corresponding largest predicted probability
	predIdxs = np.argmax(predIdxs, axis=1)

	# show a nicely formatted classification report
	print(classification_report(testGen.classes, predIdxs,
		target_names=testGen.class_indices.keys()))

	# serialize the model to disk
	print("[INFO] saving model...")
	model.save(MODEL_PATH, save_format="h5")

	print(f"[INFO] Completed {NUM_EPOCHS} epochs in {(t1-t0):.1f} sec using BATCH SIZE {BS}")
    
	# plot the training loss and accuracy
	N = NUM_EPOCHS
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
	plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
	plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
	plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
	plt.title("Training Loss and Accuracy on COVID-19 Dataset")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	plt.savefig('covid19_resnet50_plot.png')