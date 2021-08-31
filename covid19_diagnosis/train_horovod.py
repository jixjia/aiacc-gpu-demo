# USAGE
# python train.py --d dataset -p training_plot.png

# import the necessary packages
import tensorflow as tf
from tensorflow.python.ops.variables import model_variables
import horovod.tensorflow.keras as hvd
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


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default="dataset",help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="training_plot.png", help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str, default="covid19.model", help="path to output loss/accuracy plot")
ap.add_argument("-lr", "--initial_lr", type=float, default="0.001", help="initial learning rate (default 1e-3)")
ap.add_argument("-e", "--epochs", type=int, default=50, help="epoch size")
ap.add_argument("-bs", "--batch_size", type=int, default=8, help="batch size")

args = vars(ap.parse_args())

# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = args['initial_lr']
EPOCHS = args['epochs']
BATCH_SIZE = args['batch_size']

# Horovod: initialize Horovod.
hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print('[INFO] loading images...', end='')

# loop over the image paths
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

# convert the data and labels to NumPy arrays and normalize it to the range [0, 255]
data = np.array(images) / 255.0
labels = np.array(labels)

# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# partition the data into training and testing splits (80:20)
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=123)

print('[INFO] Loaded', len(trainX), 'training images and', len(testX), 'test images')

# initialize the training data augmentation object
trainAug = ImageDataGenerator(
	rotation_range=30,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# initialize the validation/testing data augmentation object (which we'll be adding mean subtraction to)
valAug = ImageDataGenerator()

# define the ImageNet mean subtraction (in RGB order) and set the mean subtraction value for each of the data augmentation objects
mean = np.array([123.68, 116.779, 103.939], dtype="float32")
trainAug.mean = mean
valAug.mean = mean

# initialize the training generator
trainGen = trainAug.flow(
	trainX, 
	trainY,
	shuffle=True,
	batch_size=BATCH_SIZE)

# initialize the validation generator
valGen = valAug.flow(
	testX,
	testY,
	shuffle=False,
	batch_size=BATCH_SIZE)

# load the VGG16 network, ensuring the head FC layer sets are left off
baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# construct the head of the model that will be placed on top of the the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(64, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become the actual model to train)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False

# compile our model
print('[INFO] compiling model...', end='')

# Horovod: adjust learning rate based on number of GPUs.
opt = Adam(lr=INIT_LR * hvd.size(), decay=INIT_LR / EPOCHS)

# Horovod: add Horovod DistributedOptimizer.
opt = hvd.DistributedOptimizer(opt)

model.compile(loss="binary_crossentropy",
                    optimizer=opt,
                    metrics=['accuracy'],
					experimental_run_tf_function=False)

callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),

    # Horovod: average metrics among workers at the end of every epoch.
    # Note: This callback must be in the list before the ReduceLROnPlateau,
    # TensorBoard or other metrics-based callbacks.
     hvd.callbacks.MetricAverageCallback(),

    # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
    # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
    # hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=3, verbose=1),
]

print('Done')

# begin fine tuning the head of the network
print(f'[INFO] GPU {hvd.rank()} -> Begin transfer learning by fine tuning the DNN head...')

# Horovod: write logs on worker 0.
t0 = time.time()
verbose = 1 if hvd.rank() == 0 else 0

history = model.fit(
			x=trainGen,
			steps_per_epoch = len(trainX) // BATCH_SIZE // hvd.size(),
			callbacks = callbacks,
			validation_data=(valGen),
			validation_steps=len(testX) // BATCH_SIZE // hvd.size(),
			epochs = EPOCHS,
			verbose = verbose)
t1 = time.time()

# execute network evaluation on head node
if hvd.rank() == 0:
	print('[INFO] evaluating network...')
	predIdxs = model.predict(testX, batch_size=BATCH_SIZE)

	# for each image in the testing set we need to find the index of the
	# label with corresponding largest predicted probability
	predIdxs = np.argmax(predIdxs, axis=1)

	# show a nicely formatted classification report
	print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))

	# compute the confusion matrix and and use it to derive the raw
	# accuracy, sensitivity, and specificity
	cm = confusion_matrix(testY.argmax(axis=1), predIdxs)
	total = sum(sum(cm))
	acc = (cm[0, 0] + cm[1, 1]) / total
	sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
	specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

	# show the confusion matrix, accuracy, sensitivity, and specificity
	print("acc: {:.4f}".format(acc))
	print("sensitivity: {:.4f}".format(sensitivity))
	print("specificity: {:.4f}".format(specificity))

	# plot the training loss and accuracy
	plot_training(history, EPOCHS, args['plot'])

	# serialize the model to disk
	print("[INFO] saving COVID-19 detector model...")
	model.save(args["model"], save_format="h5")

	print(f"[INFO] Completed {EPOCHS} epochs in {(t1-t0):.1f} sec using BATCH SIZE {BATCH_SIZE}")