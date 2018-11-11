# Behavioral Cloning

This project is in conjunction with the Udacity Self-Driving Car course.  In particular, the goal of this project is to use the Udacity car-on-track simulator to clone human steering behavior.  Recording camera views and steering angles from a simulated vehicle driven by a human provides sufficient data for us to train an autonomously steering vehicle.  The main steps are: 

* Record the training data set (not included in this repo) from a human driver.

* Use model.py to train and validate the network on this data.

* Run the model in autonomous mode in the simulator to record the performance of the cloned autonomous driver.  The final output is shown in video.mp4.

* More detail about the behavioral cloning model is provided in writeup.md.

  ![alt text][image2]

## Getting Started

### Prerequisites

Imported packages include:

```
Python
TensorFlow
Keras
sklearn
```

### Installing

No install is required --- simply clone this project from GitHub:

```
git clone https://github.com/jimwatt/P3-BehavioralCloning.git
```

## Running the Code

* Using the linux simluator provided by Udacity (not provided here), the model can be tested by running:

  `python drive.py model.h5`

  This will establish a server that listens for the simulator to provide input data.  The simulator passes in camera images as seen by the vehicle, and the model returns steering angles.


## Authors

* **James Watt**

<!--## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details-->

## Acknowledgments
This project is a submission to the Udacity Self-Driving Car nanodegree:

* https://github.com/udacity/CarND-Behavioral-Cloning-P3

The deep network architecture used here is based on the LeNet network of LeCun.

# **Behavioral Cloning** 

### Goal:

Use transfer learning to train a model that steers a simulated automobile around a track.  The model predicts correct steering angles using images of the road ahead as input.

------

**Behavioral Cloning Project**

The goals / steps of this project are the following:

- Use the simulator to collect data of good driving behavior
- Build a convolution neural network in Keras that predicts steering angles from images
- Train and validate the model with a training and validation set
- Test that the model successfully drives around track one without leaving the road
- Summarize the results with a written report

[//]: #	"Image References"
[image1]: ./images/lenet.png	"Model Visualization"
[image2]: ./images/center.jpg	"Center Camera"
[image4]: ./images/left.jpg	"Left Camera"
[image5]: ./images/right.jpg	"Right Camera"
[image3]: ./images/clockwise.jpg	"Clockwise"

## Rubric Points

#### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

------

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

- model.py containing the script to create and train the model
- drive.py for driving the car in autonomous mode
- model.h5 containing a trained convolution neural network 
- writeup_report.md (this file) summarizing the results

#### 2. Submission includes functional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of the LeNet network architecture consisting of two convolutional neural network layers with 5x5 filter sizes and depth of 6 (model.py lines 103-118) .

The model includes RELU activation layers to introduce nonlinearity (code line 111 and 113).

The data is normalized and cropped in the model using a Keras lambda layer (code line 107). 

#### 2. Attempts to reduce overfitting in the model

The model contains max-pooling layers in order to reduce the number of parameters in the model and prevent overfitting (model.py lines 112 and 114). 

Also, the training images were augmented by exploiting left-right symmetry in the steering angle and using images from different camera angles to increase the size of the data set.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 12). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road.  I used the following data sets:

- **data** : Training data provided by Udacity.
- **andrew** : Training data recorded while my son Andrew drove the simulated car three times around the track.
- **andrewright** : Training data recorded while my son Andrew drove the simulated car once around the track in the _clockwise_ direction.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to first establish the data pipeline, then enrich the data set, and then also increase the complexity of the deep network.

The main steps were as follows:

1. **Recording Data** : I first recorded data from a human driver.  I recorded data for three laps around the track.  I also recorded data from one lap of driving around the track in the clockwise direction.
2. **Combining Data** : I set up a data loading pipeline that combined my recorded data with data provided by the Udacity course.
3. **Data Pipeline** : I began by setting up an extremely simple data pipeline just to ensure that the basic pipeline for loading data, fitting a model, and then testing in the simulator was in place.  The model for this pipeline was simply a linear regression.  As expected, the performance for this pipeline was atrocious, but it at least confirmed that the data pipeline was in place.
4. **Data Preprocessing** : I introduced minimal data preprocessing using image normalization and image cropping.  These were both instituted using Lambda functions in Keras.
5. **Data Augmentation** : I also augmented the data by flipping the images from left to right to exploit symmetry in the steering angle about zero.  This effectively doubled the size of the data set.
6. **LeNet Architecture** : Next, I implemented the LeNet architecture.  This provided more parameters for the model, and training this model allowed for an improved cost and better validation accuracy.  Driving was starting to improve (the car could get around the first bend, but still wandered off the road).
7. **Further Data Augmentation** : In order to ensure that the car knew how to return to the center of the road, I decided to include the left and right camera images with adjusted steering angles.  The objective was to teach the car to return to the center of the road if it starts to veer to the side.  Using the left and right camera images effectively tripled the amount of data.  I found that I did not need to actively record human swerving behavior in order to help the autonomous vehicle return to the center of the road.
8. **AWS** : At this point, I had to move processing to the Amazon Web Service because data processing was taking too long.
9. **Generator** : Also, since memory required to store all the data was too large, I built a python "generator" to efficiently generate batches of images on the fly. 
10. **Fine Tuning** :  At this point, the trained model was starting to drive well although it still overshot one of the sharp turns.  I increased the steering correction for the left and right camera images from 0.2 to 0.25, and this did the trick to ensure that the vehicle remained in the center of the road.
11. **NVIDIA Network** : I had planned as a final step to implement the NVIDIA network to eke out further performance, although this ended up not being necessary. The LeNet architecture worked just fine.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road as seen in the movie: video.mp4.

#### 2. Final Model Architecture

The final model architecture that I used was the LeNet-5 convolutional neural network (model.py lines 105-118) consisting of two convolutional layers and three fully connected layers.  The input and output sizes of the layers are indicated in this visualization of the architecture:

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one using center lane driving. Here is an example image of center lane driving:

Center:

![alt text][image2]

I also recorded one lap of good driving behavior for the car traveling around the track in the clockwise direction:

Clockwise:

![alt text][image3]

I also used the left and right camera images here to encourage the car to remain in the center of the road.

Left:

![alt text][image4]

Right:

![alt text][image5]



To further augment the data set, I also exploited left-right symmetry in the steering angle by flipping all images. This, consequently, doubled the size of the data set.

Exploiting symmetry and using the left and right camera images effectively increased the data size _sixfold_.

After the collection process, I had 90,772 images in my training set. I kept 20% of this training set aside for validation leaving 72,618 for training and 18,154 for validation.  

Pre-processing of each image included normalization and cropping (these were performed using Lambda functions in keras).

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I observed no improvement on the validation set after three epochs.  I used an adam optimizer so that manually training the learning rate wasn't necessary.