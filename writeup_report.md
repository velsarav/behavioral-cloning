# **Behavioral Cloning** 

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output/center.jpg "center"
[image2]: ./output/right.jpg "right"
[image3]: ./output/left.jpg "left"
[image4]: ./output/cnn.png "CNN"
[image5]: ./output/validation_loss.png "Validation loss"

## Rubric Points
### The [rubric points](https://review.udacity.com/#!/rubrics/432/view) were individually addressed in the implementation and described in [code](https://github.com/velsarav/behavioral-cloning/blob/master/model.py) and [documentation](https://github.com/velsarav/behavioral-cloning/blob/master/writeup_report.md). 

---
### Project files

Project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video.mp4 video recording of vehicle driving autonomously

### Simulation

Using the Udacity provided simulator collected the image data and corresponding steering angle by manually driving the vehicle for 1 lap.

The vehicle has 3 cameras to record the images - left, center, and right.

This resulted in Total Train samples: 10242 and Total valid samples: 2562

The following are the center, right, and left images recorded at the same location:

![alt text][image1]

![alt text][image2]

![alt text][image3]

### Model Architecture 

The `model.py` file contains the code for training and saving the convolution neural network. The file shows the pipeline used for training and validating the model.

All these models are implemented using [Keras](https://keras.io/)

`get_cnn_model()` function based model consists of a convolution neural network based on [NVIDIA Architecture](https://devblogs.nvidia.com/deep-learning-self-driving-cars/)

The model used an adam optimizer, so the learning rate was not tuned manually 

### Training Strategy

#### Data Collection and classification

`read_data_csv` function read csv file from the simulated data and load line by line.

`read_img_data` function load the images using the lines path. Steering angle is for the center image. Apply correction for right(-0.2) and left image (+0.2).

#### Data Augmentation

Split the data for train and validation to solve the regression problem.

`img_brighteness` adjust the brightness based on HSV space of the image.

`generator` function randomly select center, right or left image and again randomly brighten or flip it.

`position_based_angle` function adjust the angle position based on the positive or negative value of the angle.

`adjust_steering` function appends the new angle to the corresponding images.

### Final Model Architecture

* Crop unneeded part of the image
* Resize image to a small one
* Normalize the data
* Use 5 convolutional layers
* 1 dropout layer
* 3 fully connected layer
* Optimizer Adam with learning rate 1e-5 instead of default 1e-3.

The final model architecture consisted of a convolution neural network with the following layers and layer sizes

 ![alt text][image4]

 The Total parameter is 348,219

 Train the model using `model.fit_generator()` with epochs=1 

### Model History Object
When calling `model.fit_generator()`, Keras outputs a history object that contains the training and validation loss for each epoch. to understand it better used the epochs=3 and plotted the same

 ![alt text][image5]

### Result
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

The [final video](https://github.com/velsarav/behavioral-cloning/blob/master/video.mp4) with FPS 48 shows the vehicle is able to drive autonomously around the track without leaving the road.

The car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

### Reference
* [Data Augmentation](https://medium.com/deep-learning-turkey/behavioral-cloning-udacity-self-driving-car-project-generator-bottleneck-problem-in-using-gpu-182ee407dbc5)
* [Neural Network models](https://github.com/viadanna/sdc-behaviour-cloning)