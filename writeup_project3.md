# **Traffic Sign Recognition** 

## Project 3
##### Lars Watts
##### Udacity Self-Driving Car Engineer Nanodegree
 
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1.1]: ./examples/color.png "Traffic Sign 1"
[image1.2]: ./examples/trafficSign2.jpg "Traffic Sign 2"
[image1.3]: ./examples/trafficSign3.jpg "Traffic Sign 3"
[image2]: ./examples/histogram_train.jpg "Training Histogram"
[image3]: ./examples/histogram_valid.jpg "Validation Histogram"
[image4]: ./examples/histogram_test.png "Testing Histogram"
[image_color]: ./examples/color.png "Color Sign"
[image_gray]: ./examples/grayscale.png "grayscaled sign"
[image_rotated]: ./examples/rotated.png "rotated sign"
[image_stop]: ./german_signs_resized/stop_resized.png "Stop Sign"
[image_yield]: ./german_signs_resized/yield_resized.png "Yield Sign"
[image_child]: ./german_signs_resized/children_crossing_resized.png "Children Crossing Sign"
[image_30]: ./german_signs_resized/30_resized.png "30 Sign"
[image_no_entry]: ./german_signs_resized/no_entry_resized.png "No Entry Sign"
[softmax_stop]: ./examples/softmax_results_stop2.png "Softmax results Stop Sign"
[softmax_no_entry]: ./examples/softmax_results_no_entry2.png "Softmax results No Entry"
[softmax_yield]: ./examples/softmax_results_yield2.png "Softmax results Yeild"
[softmax_30]: ./examples/softmax_results_30kp2h.png "Softmax results 30 km/h"
[softmax_child]: ./examples/softmax_results_child2.png "Softmax results Children Crossing"



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/lswatts42/carnd_project3_TrafficSignClassification/blob/master/Traffic_Sign_Classifier-Copy1.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the python library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34779 samples
* The size of the validation set is 4410 samples
* The size of test set is 12630 samples
* The shape of a traffic sign image is 32 x 32 x 3 pixels
* The number of unique classes/labels in the data set is 42

#### 2. Include an exploratory visualization of the dataset.

The dataset includes thousands of pictures of traffic signs taken on German roads. Here is some examples of the pictures in the datasets:

![Traffic Sign 1][image1.1]
![Traffic Sign 2][image1.2]
![Traffic Sign 3][image1.3]

These are histograms of the training, validation, and testing datasets, where the x axis is the classification of the traffic sign and the y axis is how many instances of that sign there are in the dataset. Notice how they all have a very similar distribution of signs, and that some signs appear far more often than the others. 

![Histogram of the Training dataset][image2]
![Histogram of the Validation dataset][image3]
![Histogram of the Testing dataset][image4]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)


1. Grayscale
First, I converted the images to grayscale. I did this with the following code:
`gray = np.sum(X_train_new/3, axis = 3, keepdims=True)`
Although this is not the exact equation that converts a RGB file to grayscale, it works. I tested both and there was no change in performance. 
2. Augment Dataset
Then, I augmented the dataset by appending a rotated version of the original dataset to the end. I experimented with a 20, 90, and 180 degrees of rotation and found that 180 degrees worked best. This doubled the amount of data that I could use to train the algorithm with hardly any effort on my part. It's interesting that even though the algorithm doesn't see any rotated signs in the validation or test dataset, this still improves the performance of the network significantly. Perhaps it is because the signs are, in general, quite symmetric. 
3. Normalize the dataset
Finally, I normalized the dataset. I did this by dividing all the pixel values by 255, which is the maximum pixel value possible. The end result was that all of the numbers were a decimal between 0 and 1, which helps with the training process. 

Here is an example of a traffic sign image before and after grayscaling:

![color][image_color]![gray][image_gray]

Here is the same image after being rotated 180 degrees:

![rotated][image_rotated]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x16 	|
| RELU6					|												|
| Convolution 3x3       | 1x1 stride, valid padding, outputs 28x28x64   |
| RELU6                 |                                               |
| Max pooling	      	| 2x2 stride,  outputs 14x14x64 (POOL1)			|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x128  |						| RELU6                 |                                               |
| Max Pooling           | 2x2 stride, outputs 5x5x128                   |
| Flatten               | Flatten result, then append flattened POOL1   |
| Fully connected		| input: 17600, output: 320						|
| RELU6                 |                                               |
| Dropout layer         |                                               |
| Fully connected       | input: 320, output: 84                        |
| RELU6                 |                                               |
| Dropout               |                                               |
| Fully Connected       | input: 84, output: 42                         |
| Softmax				|           									|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an optimizer to minimize a loss function. The loss function consisted of the cross entropy of the logits and the labels. I also included L2 regularization in the loss function to eliminate overfitting, and 0.01 was the multiplier I used for each calculation. The optimizer I used was the Adam optimizer, which uses stochastic gradient descent to find minima. The number of Epochs I settled with was 12, though it often reached the max accuracy by epoch 10. Learning rate was set to 0.001, and the batch size was 128. I tested various learning rates, and found that if I set it too high, then it tended to resort to overfitting quikly, but if I set it too low, it would take too long to train. I also tested various batch sizes, with a negligible amount of change in the results. 
Another way I discouraged overfitting was by incorporating dropout in the training process. I started with 50% dropout, then settled with 86% in the end. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.2%
* validation set accuracy of 95.3% 
* test set accuracy of 93.0%

I started with the LeNet architecture and set about improving it from there. I chose LeNet because it does well with small images and is a very lightweight model with few parameters. The initial accurcy was about 88-90%. I started with color images as well, though I later switched to grayscale and achieved a slight improvement in performance and training time. The first changes I made were inspired by [this paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) by Pierre Sermanet and Yann Lecun. They took the output of the first pooling layer and added it to the output of the second pooling layer before it went through the fully connected layer. I did the same thing, though the one difference was that I have multiple fully connected layers, so I chose the first one. 
Another change I made was to the activation function. After a little bit of research, I found that the Relu6 function was a modern version of the Relu function and increased the performance. According to [this paper](http://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf), the Relu6 function "encourages the model to learn sparse features earlier" than the normal Relu activation function. 
Further research told me to make the model "deeper" by adding layers. So I added a convolutional layer to the beginning (with "same" padding), and then another fully connected layer near the end. These two additions improved the performance significantly. Before I decided on just two added layers, I also tested having more filters in each layer, and adding even more layers. Those tactics did not help, and sometimes even hurt the accuracy of the model. Not only that, but training the models took an extremely long time due to the immense number of parameters that had to be tuned.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found using an web image search:

![stop][image_stop] ![no entry][image_no_entry] ![yield][image_yield] 
![30 kph][image_30] ![child][image_child]

I picked these because they were a mix of difficult to predict signs (like the children crossing sign) and normal signs (stop, yield, speed limit). 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction   					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   							| 
| No Entry     			| No Entry 								|
| Yield					| Yield	 								|
| 30 km/h	      		| 30 km/h  				 				|
| Children Crossing		| Children Crossing   					|


The model was able to correctly all 5 of the traffic signs, which gives an accuracy of 100%. This makes sense to me, because the pictures I used were all in sunny, bright conditions whereas the test dataset had a lot of darker images. But even then, I'm sure my accuracy would decrease if I had more samples, and it would get close to 94%, which was the result from the main part of the project. I ran it a few more times, and half of the time it got 80% accuracy, and the other half it was perfect. It seems to depend on the end result of the training. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Each of the decisions that the model made on the new signs was made with very high certainty, at least 95%. Below are the bar graphs for each of the signs, with the top 5 softmax probabilities and their corresponding signs. 

##### Stop Sign:
![stop softmax][softmax_stop]

##### No Entry Sign:
![stop no entry][softmax_no_entry]

##### Yield Sign:
![stop yield][softmax_yield]

##### 30 km/h Sign:
![stop 30][softmax_30]

##### Children Crossing Sign:
![stop child][softmax_child]