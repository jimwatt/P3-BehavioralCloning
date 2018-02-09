# **Behavioral Cloning** 

### Goal:

Use transfer learning to train a model that steers a simulated automobile around a track.  The model predicts correct steering angles using images of the road ahead as  input.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # "Image References"

[image1]: ./images/lenet.png "Model Visualization"
[image2]: ./images/center.jpg "Center Camera"
[image4]: ./images/left.jpg "Left Camera"
[image5]: ./images/right.jpg "Right Camera"
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

My model consists of the LeNet architecture consisting of a convolution neural network with 5x5 filter sizes and depth of 6 (model.py lines 103-118) .

The model includes RELU activation layers to introduce nonlinearity (code line 111 and 113), and the data is normalized in the model using a Keras lambda layer (code line 107). 

#### 2. Attempts to reduce overfitting in the model

The model contains max-pooling layers in order to reduce overfitting (model.py lines 112 and 114). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 12). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road.  I used the following data sets:

* **data** : Training data provided by Udacity.
* **andrew** : Training data recorded while my son Andrew drove the simulated car three times around the track.
* **andrewright** : Training data recorded while my son Andrew drove the simulated car once around the track in the clockwise direction.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to 

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

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

Left:

![alt text][image4]

Right:

![alt text][image5]



To further augment the data set, I also exploited left-right symmetry in the steering angle by flipping all images. This, consequently, doubled the size of the data set.

After the collection process, I had 90,772 images in my training set. I kept 20% of this training set aside for validation leaving 72,618 for training and 18,154 for validation.  

Pre-processing of each image included normalization and cropping (these were performed using Lambda functions in keras).

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I observed no improvement on the validation set after three epochs.  I used an adam optimizer so that manually training the learning rate wasn't necessary.