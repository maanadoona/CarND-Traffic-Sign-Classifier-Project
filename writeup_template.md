# **Traffic Sign Recognition** 

## Project : Traffic Sign Classifier

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[./data/]: # (Image References for Training)
[./mydata/]: # (Image References for Pridiction and Top5)

[image1]: ./result/training/0.png "Training Data"
[image1-1]: ./result/training/1.png "Training Data"
[image1-2]: ./result/training/2.png "Training Data"
[image1-3]: ./result/training/3.png "Training Data"
[image1-4]: ./result/training/4.png "Training Data"
[image1-5]: ./result/training/5.png "Training Data"
[image1-6]: ./result/training/6.png "Training Data"
[image1-7]: ./result/training/7.png "Training Data"
[image1-8]: ./result/training/8.png "Training Data"
[image1-9]: ./result/training/9.png "Training Data"

[image2-00]: ./result/preprocessing/0_0.png "Training Data"
[image2-01]: ./result/preprocessing/0_1.png "Gray Data"
[image2-02]: ./result/preprocessing/0_2.png "Normalization Data"
[image2-10]: ./result/preprocessing/1_0.png "Training Data"
[image2-11]: ./result/preprocessing/1_1.png "Gray Data"
[image2-12]: ./result/preprocessing/1_2.png "Normalization Data"
[image2-20]: ./result/preprocessing/2_0.png "Training Data"
[image2-21]: ./result/preprocessing/2_1.png "Gray Data"
[image2-22]: ./result/preprocessing/2_2.png "Normalization Data"
[image2-30]: ./result/preprocessing/3_0.png "Training Data"
[image2-31]: ./result/preprocessing/3_1.png "Gray Data"
[image2-32]: ./result/preprocessing/3_2.png "Normalization Data"
[image2-40]: ./result/preprocessing/4_0.png "Training Data"
[image2-41]: ./result/preprocessing/4_1.png "Gray Data"
[image2-42]: ./result/preprocessing/4_2.png "Normalization Data"

[image3-0]: ./result/testmodel/0.jpg "My data"
[image3-1]: ./result/testmodel/1.jpg "My data"
[image3-2]: ./result/testmodel/2.jpg "My data"
[image3-3]: ./result/testmodel/3.jpg "My data"
[image3-4]: ./result/testmodel/4.jpg "My data"
[image3-5]: ./result/testmodel/5.jpg "My data"
[image3-6]: ./result/testmodel/6.jpg "My data"
[image3-7]: ./result/testmodel/7.jpg "My data"
[image3-8]: ./result/testmodel/8.jpg "My data"
[image3-9]: ./result/testmodel/9.jpg "My data"

[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/maanadoona/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]![alt text][image1-1]![alt text][image1-2]![alt text][image1-3]![alt text][image1-4]
![alt text][image1-5]![alt text][image1-6]![alt text][image1-7]![alt text][image1-8]![alt text][image1-9]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data.
I chose two techniques to get improved recognition result.
1. Converting to grayscale
 - Conversion helps us to reduce computation and algorithms
 
2. Normalization
 - normalization avoids the influense of high frequency noise and very low noise.

Here is an example of a traffic sign image before and after grayscaling and after normalization.

![alt text][image2-00]![alt text][image2-01]![alt text][image2-02]
![alt text][image2-10]![alt text][image2-11]![alt text][image2-12]
![alt text][image2-20]![alt text][image2-21]![alt text][image2-22]
![alt text][image2-30]![alt text][image2-31]![alt text][image2-32]
![alt text][image2-40]![alt text][image2-41]![alt text][image2-42]

As a last step, I normalized the image data because this can avoid the influence of high frequency noise and very low noise. Actually, I get the different result(reduced the performance as -2%) when I exclude this technique.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My first model consitsted of the following layers:
| Layer   1      		|     Description	   Convolutional     					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 gray & normalized image   							| 
| Convolution 1   	| 1x1 stride, same padding, outputs 28x28x12 	|
| RELU 1					|												|
| Max pooling	 1   	| 2x2 stride,  outputs 14x14x12 				|

| Layer   2      		|     Description	   Convolutional     					| 
| Input         		| 14x14x12 image   							| 
| Convolution 2	    | 2x2 stride, outputs 10x10x16   									|
| RELU 1					|												|
| Max pooling	 1   	| 2x2 stride,  outputs 5x5x25 				|

| Flatten	 1   	| inputs 5x5x25,  outputs 625 				|

| Layer   3      		|     Description	   Dropout & Fully Connected.    					| 
| Input         		| 625 image   							| 
| Fully Connected 1	    | outputs 300   									|

| Layer   4      		|     Description	   Fully Connected.    					| 
| Input         		| 300 image   							| 
| Fully Connected 2	    | outputs 100   									|

| Layer   5      		|     Description	   Fully Connected.    					| 
| Input         		| 100 image   							| 
| Fully Connected 2	    | outputs 43   									| 

My final model consisted of the following layers:
| Layer   1      		|     Description	   Convolutional     					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 gray & normalized image   							| 
| Convolution 1   	| 1x1 stride, same padding, outputs 28x28x12 	|
| RELU 1					|												|
| Max pooling	 1   	| 2x2 stride,  outputs 14x14x12 				|

| Layer   2      		|     Description	   Convolutional     					| 
| Input         		| 14x14x12 image   							| 
| Convolution 2	    | 1x1 stride, outputs 10x10x16   									|
| RELU 1					|												|
| Max pooling	 1   	| 2x2 stride,  outputs 5x5x16 				|

| Flatten	 1   	| inputs 5x5x16,  outputs 400 				|

| Layer   3      		|     Description	   Dropout & Fully Connected.    					| 
| Input         		| 400 image   							| 
| Fully Connected 1	    | outputs 200   									|

| Layer   4      		|     Description	   Fully Connected.    					| 
| Input         		| 200 image   							| 
| Fully Connected 2	    | outputs 100   									|

| Layer   5      		|     Description	   Fully Connected.    					| 
| Input         		| 100 image   							| 
| Fully Connected 2	    | outputs 43   									| 

I used the model like Lenet for the first time.
I got the result of 0.91 on average.
I modified the model to enhance the performance over 0.93.
I finalized got the model upperside.

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an such as optimizer like below :
batch size = 32
epochs = 10
learning rate = 0.001


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
EPOCH 1/10 ...
Validation Accuracy = 0.901
EPOCH 2/10 ...
Validation Accuracy = 0.949
EPOCH 3/10 ...
Validation Accuracy = 0.965
EPOCH 4/10 ...
Validation Accuracy = 0.973
EPOCH 5/10 ...
Validation Accuracy = 0.982
EPOCH 6/10 ...
Validation Accuracy = 0.984
EPOCH 7/10 ...
Validation Accuracy = 0.982
EPOCH 8/10 ...
Validation Accuracy = 0.988
EPOCH 9/10 ...
Validation Accuracy = 0.989
EPOCH 10/10 ...
Validation Accuracy = 0.988

* validation set accuracy of 0.956
* test set accuracy of 0.9377672208931742

If an iterative approach was chosen: 
 I chode the first architecture of Lenet but I cannot get the accuracy of 0.93 over.
 So, I modified the paramaters to improve the performace up.
 First, I forcused on pre-processing. I missed the normalization. It makes the performace lower about -2 or -3 percents. So I found that normalization is essential.
 
 Actually, EPOCH is important factor to get the performace, but I cannot wait the total time for calculation. So I fixed the EPOCH to just 10.
 
 To up the performace of 0.93, I modified the parameters in layers because lenet algorithm's performace is near to 0.93 but test procedure get the below 0.93.
 I tuend the paramaters of layer2's depth from 25 to 16 and after layer 3, I reduce the size in half.
 
 Finally, I got the performance of 0.937 on test case.
 
 Of course, I think it's not the optimized result but I had the several time of test and I checked the even if it has the result the lowest but it's over than 0.93.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image3-0]![alt text][image3-1]![alt text][image3-2]![alt text][image3-3]![alt text][image3-4]
![alt text][image3-5]![alt text][image3-6]![alt text][image3-7]![alt text][image3-8]![alt text][image3-9]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image		 Yield 		      |     Prediction	 1.0     					| 
| Yield      		|  1.0   									| 
| Priotiry road      		|  0.947e-21   									| 
| Road work     		| 1.24e-21 										|
| Traffic signals 		| 7.09e-22 											|
| Roundabout mandatory  		| 2.24e-23					 				|

| Image		 Vehicles over 3.5 metric tons prohibited 		      |     Prediction	 0.82     					| 
| Vehicles over 3.5 metric tons prohibited      		|  0.82   									| 
| Ebd if bi oassubg      		|  0.167   									| 
| Priority road     		| 0.013 										|
| Roundabout mandatory  		| 0.00059 											|
| No passing  		| 1.06e-05 					 				|

I wrote the results of just 2. You can see the result the html files others.
The model was able to correctly guess 10 of the 10 traffic signs, which gives an accuracy of 100%.
In fact, I tried several times I got the result of 8 ~ 10. It depends on try and errors.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.
For the first image, the model is relatively sure that this is a stop sign (probability of 1.0), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	   1.0     					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Yield   									| 
| .82     				| Vehicles over 3.5 metric tons prohibited 										|
| 1.0       			| Slippery road											|
| .998      			| Roundabout mandatory					 				|
| 1.0       		    | Ahead only      							|
| 1.0       		    | Bumpy road      							|
| 1.0       		    | General caution     							|
| 1.0       		    | Stop      							|
| 0.633       		    | Speed limit (100km/.h)      							|
| 1.0       		    | No entry      							|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


