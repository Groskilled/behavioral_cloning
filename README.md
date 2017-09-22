#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center_driving.jpg "driving center"
[image2]: ./examples/recover_left.jpg "recover from left"
[image3]: ./examples/recover_right.jpg "recover from right"
[image4]: ./examples/flip.png "flipped image"
[image5]: ./examples/center_driving.jpg "Model Visualization"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths of 4 to 16 (model.py lines 10-17) 

The model's layers includes RELU activation to introduce nonlinearity (code line 10), the data is not normalized because I could not get the model working in autonomous mod (unknown instruction, core dump...). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (main.py line 27). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (main.py line 27).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, avoiding dirt road and water. 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to get something looking like a simplified vgg16 so I could run it on my laptop.

My first step was to use a convolution neural network model similar to the vgg16. I thought this model might be appropriate because it is often used when dealing with images.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had problems to follow the road when the side of the road was not a yellow line or a the red and wihte stripes. 

To deal with this issue I recorded some examples where the car goes from a look alike position back to the middle of the road. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 8-24) consisted of a convolution neural network with the following layers and layer sizes: 2 conv2d layers with a 3x3 kernel size followed by a maxpooling (2x2) layer (let's call that a block), and the outuput size of the conv layers is doubled in each block (4->8->16). This is followed by a Flatten layer then 2 Dense layer of 120 and 80 neurons with a dropout probability of 50%.

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to stay away from the sides of the road. These images show what a recovery looks like:

![alt text][image2]
![alt text][image3]

To augment the data sat, I also flipped images and angles thinking that this would help get a bigger dataset without recording more. For example, here is an image that has then been flipped:

![alt text][image3]
![alt text][image4]


After the collection process, I had 10000 number of data points. I then preprocessed this data by flipping it and using the left and right camera image.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by the validation and training loss which did not move enough after this epoch. I used an adam optimizer so that manually training the learning rate wasn't necessary.
