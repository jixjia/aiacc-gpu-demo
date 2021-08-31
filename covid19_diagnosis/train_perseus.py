# import the necessary packages
import perseus.tensorflow.horovod.keras as hvd
import tensorflow as tf
from tensorflow.python.ops.variables import model_variables
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
import time
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="dataset",help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="training_plot.png", help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str, default="covid19.model", help="path to output loss/accuracy plot")
ap.add_argument("-lr", "--initial_lr", type=float, default=0.001, help="initial learning rate (default 1e-3)")
ap.add_argument("-e", "--epochs", type=int, default=50, help="epoch size")
ap.add_argument("-bs", "--batch_size", type=int, default=8, help="batch size")
args = vars(ap.parse_args())


def plot_training(history, N, plotPath):
	# construct a plot that plots and saves the training history
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
	plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
	plt.plot(np.arange(0, N), history.history["accuracy"], label="train_acc")
	plt.plot(np.arange(0, N), history.history["val_accuracy"], label="val_acc")
	plt.title("Training Loss and Accuracy on COVID-19 Dataset")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	plt.savefig(plotPath)


def data_augmentation(trainX,testX,trainY,testY,batch_size):
	# initialize the training data augmentation
	trainGen = ImageDataGenerator(
		rotation_range=30,
		zoom_range=0.15,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.15,
		horizontal_flip=True,
		fill_mode="nearest"
		)

	# initialize the validation/testing data augmentation
	valGen = ImageDataGenerator()

	# set ImageNet mean subtraction (in RGB order) and apply it to the mean subtraction value for each of the data augmentation objects
	mean = np.array([123.68, 116.779, 103.939], dtype="float32")
	trainGen.mean = mean
	valGen.mean = mean

	# initialize the training generator
	trainAug = trainGen.flow(
		trainX, 
		trainY,
		shuffle=True,
		batch_size=batch_size)

	# initialize the validation generator
	valAug = valGen.flow(
		testX,
		testY,
		shuffle=False,
		batch_size=batch_size)
	
	return trainAug, valAug


def construct_model():
	# load VGG network with pre-trained ImageNet, lay off head FC layer
	baseModel = VGG16(weights="imagenet", 
					  include_top=False, 
					  input_tensor=Input(shape=(224, 224, 3)))

	# construct new head for fine-tuning
	headModel = baseModel.output
	headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
	headModel = Flatten(name="flatten")(headModel)
	headModel = Dense(64, activation="relu")(headModel)
	headModel = Dropout(0.5)(headModel)
	headModel = Dense(2, activation="softmax")(headModel)

	# place the head FC model on top of the base model
	model = Model(inputs=baseModel.input, outputs=headModel)

	# freeze base model layers
	for layer in baseModel.layers:
		layer.trainable = False

	# compile model: Horovod
	opt = Adam(lr=INIT_LR * hvd.size(), decay=INIT_LR / EPOCHS)
	
	opt = hvd.DistributedOptimizer(opt)

	model.compile(loss="binary_crossentropy", 
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

	return model, callbacks


def serialize_model(model, outputPath):
	print("[INFO] saving COVID-19 diagnostic model...")
	model.save(outputPath, save_format="h5")


# initialize Horovod
hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


# initialize the initial LR, number of epochs and pool batch size
INIT_LR = args['initial_lr']
EPOCHS = args['epochs']
BATCH_SIZE = args['batch_size']

# load images
print('[INFO] loading images...', end='')

imagePaths = list(paths.list_images(args["dataset"]))
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

print('Done')

# convert to np arrays and normalize it to rgb range [0, 255]
data = np.array(images) / 255.0
labels = np.array(labels)

# one-hot encode labels by directory name
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# train/test split (80:20)
(trainX, testX, trainY, testY) = train_test_split(data, 
												  labels,
												  test_size=0.20, 
												  stratify=labels, 
												  random_state=123)

# data augmentation
trainAug, valAug = data_augmentation(trainX,testX,trainY,testY,BATCH_SIZE)

# construct model
print('[INFO] compiling model...', end='')
model, callbacks = construct_model()
print('Done')

# fine-tune the network
print(f'[INFO] GPU {hvd.rank()} -> Transfer learning by fine tuning the head layer..')

verbose = 1 if hvd.rank() == 0 else 0
t0 = time.time()
history = model.fit(
			x=trainAug,
			steps_per_epoch = len(trainX) // BATCH_SIZE // hvd.size(),
			validation_data=(valAug),
			validation_steps=len(testX) // BATCH_SIZE // hvd.size(),
			epochs = EPOCHS,
			verbose = verbose,
			callbacks = callbacks)
t1 = time.time()

if hvd.rank() == 0:
	# serialize the model to disk
	serialize_model(model, args["model"])

	# performance evaluation
	print('[INFO] evaluating model...')
	predIdxs = model.predict(testX, batch_size=BATCH_SIZE)
	predIdxs = np.argmax(predIdxs, axis=1)

	# show the confusion matrix, accuracy, sensitivity, and specificity
	cm = confusion_matrix(testY.argmax(axis=1), predIdxs)
	total = sum(sum(cm))
	acc = (cm[0, 0] + cm[1, 1]) / total
	sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
	specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

	print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))
	print("acc: {:.4f}".format(acc))
	print("sensitivity: {:.4f}".format(sensitivity))
	print("specificity: {:.4f}".format(specificity))

	# plot training loss and accuracy
	plot_training(history, EPOCHS, args['plot'])

	print(f"[INFO] Completed {EPOCHS} epochs in {(t1-t0):.1f} sec using BATCH SIZE {BATCH_SIZE}")