# Behavioral Cloning

This project is in conjunction with the Udacity Self-Driving Car course.  In particular, the goal of this project is to use the Udacity car-on-track simulator to clone human steering behavior.  Recording camera views and steering angles from a simulated vehicle driven by a human provides sufficient data for us to train an autonomously steering vehicle.  The main steps are: 

* Record the training data set (not included in this repo) from a human driver.
* Use model.py to train and validate the network on this data.
* Run the model in autonomous mode in the simulator to record the performance of the cloned autonomous driver.  The final output is shown in video.mp4.
* More detail about the behavioral cloning model is provided in writeup.md.

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