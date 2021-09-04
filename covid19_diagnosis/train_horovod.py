# mpirun -np 4 --allow-run-as-root python train_perseus.py -e 10 -s 10 -bs 4

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
import horovod.tensorflow.keras as hvd
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
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import time

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


def data_augmentation(train_path, val_path, test_path, batch_size):
	# initialize the training data augmentation
	trainAug = ImageDataGenerator(
		rotation_range=25,
		zoom_range=0.1,
		width_shift_range=0.1,
		height_shift_range=0.1,
		shear_range=0.2,
		horizontal_flip=True,
		fill_mode="nearest")

	# initialize the validation/testing data augmentation
	valAug = ImageDataGenerator()

	# set ImageNet mean subtraction (in RGB order) and apply it to the mean subtraction value for each of the data augmentation objects
	mean = np.array([123.68, 116.779, 103.939], dtype="float32")
	trainAug.mean = mean
	valAug.mean = mean

	# initialize the training generator
	trainGen = trainAug.flow_from_directory(
		train_path,
		class_mode="categorical",
		target_size=(224, 224),
		color_mode="rgb",
		shuffle=True,
		batch_size=batch_size)

	# initialize the validation generator
	valGen = valAug.flow_from_directory(
		val_path,
		class_mode="categorical",
		target_size=(224, 224),
		color_mode="rgb",
		shuffle=False,
		batch_size=batch_size)

	# initialize the testing generator
	testGen = valAug.flow_from_directory(
		test_path,
		class_mode="categorical",
		target_size=(224, 224),
		color_mode="rgb",
		shuffle=False,
		batch_size=batch_size)
	
	return trainGen, valGen, testGen


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
		hvd.callbacks.MetricAverageCallback(),
		# hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=3, verbose=1),
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

totalTrain = len(list(paths.list_images(TRAIN_PATH)))
totalVal = len(list(paths.list_images(VAL_PATH)))
totalTest = len(list(paths.list_images(TEST_PATH)))

print(f'[INFO] GPU {hvd.rank()} -> \nTrainset: {totalTrain}\nValset: {totalVal}\nTestset: {totalTest}')

# data augmentation
trainGen, valGen, testGen = data_augmentation(TRAIN_PATH, VAL_PATH, TEST_PATH, BS)

# construct model
print(f'[INFO] GPU {hvd.rank()} -> preparing model...')
model, callbacks = construct_model(INIT_LR, NUM_EPOCHS)

# fine-tune the network
print(f'[INFO] GPU {hvd.rank()} -> training model...')

t0 = time.time()
H = model.fit_generator(
	trainGen,
	steps_per_epoch = STEP_SIZE if STEP_SIZE else totalTrain // BS  // hvd.size(),
	# validation_data = valGen,
	# validation_steps = totalVal // hvd.size(),
	epochs = NUM_EPOCHS,
	verbose = 1 if hvd.rank() == 0 else 0,
	callbacks = callbacks)
t1 = time.time()

if hvd.rank() == 0:
	# serialize the model to disk
	serialize_model(model, MODEL_PATH)

	# performance evaluation
	print('[INFO] evaluating model...')
	testGen.reset()
	predIdxs = model.predict_generator(testGen, steps=(totalTest // BS) + 1)
	predIdxs = np.argmax(predIdxs, axis=1)

	# show a nicely formatted classification report
	print(classification_report(testGen.classes, predIdxs,target_names=testGen.class_indices.keys()))

	# plot training loss and accuracy
	plot_training(H, NUM_EPOCHS, PLOT_PATH)

	# summary
	print(f"[INFO] Completed {NUM_EPOCHS} epochs in {(t1-t0):.1f} sec using BATCH SIZE {BS}")