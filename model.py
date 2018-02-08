import csv
import cv2
import numpy as np

#################################################################
# 1. Load and pre-process the data
#################################################################
# Specifiy the directories where we can find data
# data : training data provided by Udacity
# andrew : three laps counter-clockwise around track
# andrewright : one lap clockwise around the track
datadirs = ['../data/','../andrew/','../andrewright/']

# First, get every line from the log files we are going to consider, along with its image directory
samples = []
for datadir in datadirs:

	print("Loading data from {} ".format(datadir))

	# Read in the csv driving log file
	datfile = datadir + 'driving_log.csv'
	imgdir = datadir + 'IMG/'
	with open(datfile, 'r') as f:
	        reader = csv.reader(f)
	        next(reader)	# skip the header line
	        for line in reader:
	        	samples.append((line,imgdir))	# store the line and the image directory

# split the samples into train and validation sets
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

################################################################
# 2. Define the generator for producing data efficiently in memory
################################################################
def generator(samples, batch_size):
	numsamples = len(samples)
	while True:	
		shuffle(samples)
		for offset in range(0,numsamples,batch_size):	# for each batch of samples from the log files
			batch_samples = samples[offset:offset+batch_size]
			images = []
			measurements = []

			for line,imgdir in batch_samples:		# for each sample in the batch
				steering_center = float(line[3])	# grab the steering value
				
				# Augment the data using the left and right side cameras.  
				# (Using left and right cameras helps the vehicle learn to hold to the center of the road.)
				# Create adjusted steering measurements for the side camera images
				correction = 0.25 # this is a parameter to tune
				steering_left = steering_center + correction
				steering_right = steering_center - correction

				# read in images from center, left and right cameras
				filename = line[0].split('/')[-1]
				img_center = cv2.imread(imgdir+filename)
				img_center = cv2.cvtColor(img_center, cv2.COLOR_BGR2RGB)

				filename = line[1].split('/')[-1]
				img_left = cv2.imread(imgdir+filename)
				img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)

				filename = line[2].split('/')[-1]
				img_right = cv2.imread(imgdir+filename)
				img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)

				# add all images and angles to data set
				images.append(img_center)
				images.append(img_left)
				images.append(img_right)
				measurements.append(steering_center)
				measurements.append(steering_left)
				measurements.append(steering_right)

			# Further data augmentation
			# For every image add the flipped version of the image
			augmented_images, augmented_measurements = [], []
			for image,measurement in zip(images,measurements):
				augmented_images.append(image)
				augmented_measurements.append(measurement)
				augmented_images.append(cv2.flip(image,1))
				augmented_measurements.append(measurement*-1.0)

			# Convert image list to numpy arrays
			X_train = np.array(augmented_images)
			y_train = np.array(augmented_measurements)

			# Note:  for batch of size batch_size, we actually get a batch of size 6*batch_size.  3 camera views x 2 flip
			yield shuffle(X_train, y_train)

# Set the batch size 
batch_size = 32  # We will actually have 6*batch_size images in the batch because of data augmentation

# Create the training and validation generators
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

#######################################################################
# 3. Define the model network architecture
#######################################################################
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Cropping2D
model = Sequential()
# This layer simply performs image Normalization
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
# This layer crops the image
model.add(Cropping2D(cropping=((70,25),(0,0))))
# LeNet Network (CNNs with max-pooling followed by densely connected layers)
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

###########################################################################
# 4. Train the network using the data.
###########################################################################
print("training ...")
model.compile(loss='mse', optimizer='adam')		# use the Mean-square error cost, and the adam optimizer
model.fit_generator(generator=train_generator, samples_per_epoch=len(train_samples)*6, 
validation_data=validation_generator, nb_val_samples=len(validation_samples)*6, nb_epoch=3)

###########################################################################
# 5. Save the model.
###########################################################################
model.save('model.h5')
