# **Traffic Sign Recognition via Deep Convolutional Neural Network ** 

## Writeup

### by Zikri Bayraktar, Ph.D.

---

**Build a Traffic Sign Recognition Classifier Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/histogram_of_input.png "Histogram"
[image2]: ./examples/Sample_Signs.png "Sample Signs"
[image3]: ./examples/Data_Augmentation.png "Data Augmentation"
[image4]: ./examples/LeNET.png "LeNET"
[image5]: ./examples/New_Images_Normalized.png "New Images Normalized"
[image6]: ./examples/New_Image_01.png "New Sign 01"
[image7]: ./examples/New_Image_02.png "New Sign 02"
[image8]: ./examples/New_Image_03.png "New Sign 03"
[image9]: ./examples/New_Image_04.png "New Sign 04"
[image10]: ./examples/New_Image_05.png "New Sign 05"
[image11]: ./examples/New_Image_06.png "New Sign 06"
[image12]: ./examples/New_Image_07.png "New Sign 07"
[image13]: ./examples/New_Image_08.png "New Sign 08"
[image14]: ./examples/New_Image_09.png "New Sign 09"
[image15]: ./examples/New_Image_10.png "New Sign 10"
[image16]: ./examples/New_Image_11.png "New Sign 11"
[image17]: ./examples/New_Image_12.png "New Sign 12"
[image18]: ./examples/New_Image_13.png "New Sign 13"
[image19]: ./examples/New_Image_14.png "New Sign 14"



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Here is the link to my [project code](./Traffic_Sign_Classifier.ipynb).  I completed this work on AWS GPU g2-2xlarge instances.

### Data Set Summary & Exploration

#### 1. [German Traffic Signs Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) was collected and resized by other researchers and we are provided with pickle files of dictionaries with keys of 'features', 'labels', 'sizes' and 'coords'.
Data was pre-split into three (3) Pickle files; one for training, one for validation and one for testing. 

I used simple python commands to figure out the shape of the data sets summarized as follows:

* The size of training set is 34799 examples
* The size of the validation set is 4410 examples
* The size of test set is 12630 examples
* The shape of a traffic sign image is 32x32
* The number of unique classes/labels in the data set is 43.

Code block for this is as follows:
```python
# TODO: Number of training examples
n_train = X_train.shape[0]
# TODO: Number of validation examples
n_validation = X_valid.shape[0]
# TODO: Number of testing examples.
n_test = X_test.shape[0]
# TODO: What's the shape of an traffic sign image?
image_shape = (X_train.shape[1],X_train.shape[2])
# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(np.concatenate((y_train, y_valid, y_test), axis=0)))

```

#### 2. Exploratory analysis is important step in understanding the class imbalance in the data sets. Using the 'collections.Counter' command, we can figure out how many examples there are in each datasets. 
Here is an exploratory histogram visualization of the data set showing how each label is distributed. There is a clear class imbalance for all three datasets.

![alt text][image1]

I also plotted a handful of the input images to view what kind of inputs that we are dealing with. Couple of things jump out. (1) Illumination is different from image to image. (2) Same sign may have zoomed or rotated. (3) There might be other background information other than the sign itself.
All of these observations can help us to decide how to augment the input data sets.
![alt_text][image2]

Code block for exploratory analysis is below:
```python
# How many examples per class:
print('How many examples per class in trainig dataset:')
print(Counter(y_train))

fig=plt.figure(figsize=(12,3))
for i in range(1,15):
    index = random.randint(0, len(X_train))
    imageN = X_train[index].squeeze()
    fig.add_subplot(2, 7, i)
    plt.imshow(imageN)
plt.show()

# Plot the histogram of class labels:
plt.figure(figsize=(10,4))
plt.hist([y_test, y_train, y_valid], bins=43)
plt.title('Histogram of Classes')
plt.legend(['Test', 'Train', 'Validation'])
```

### Design and Test a Model Architecture

#### As a first step, I normalized so that the data has mean zero and equal variance. For image data, (pixel - 128)/ 128 is a quick way to approximately normalize the data and used in this project. Then, I rotated the image 90, 180, 270 degrees. Finally, I added 5% noise to the normalized dataset.

Here is an example of a traffic sign image on the left along with the augmented images. Second image is the normalized one, third one is rotated 90 degrees, fourth one is rotated 180 degrees, fifth one is rotated 270 degrees and the final one is noise added.

![alt text][image3]

Code block for data augmentation is below:
```python
from sklearn.utils import shuffle
import skimage.transform

# Normalize the training data:
for i in range(X_train.shape[0]):
    X_trainN[i] = normalizeImage(X_train[i])

# Normalize the validation set:
for i in range(X_valid.shape[0]):
    X_validN[i] = normalizeImage(X_valid[i])

# Normalize the test set:
for i in range(X_test.shape[0]):
    X_testN[i] = normalizeImage(X_test[i])

# Rotate 90/180/270 degrees:
for i in range(X_train.shape[0]):
    X_train90[i]  = skimage.transform.rotate(X_trainN[i], 90)
    X_train180[i] = skimage.transform.rotate(X_trainN[i], 180)
    X_train270[i] = skimage.transform.rotate(X_trainN[i], 270)
    
# Add 5% noise:
for i in range(X_train.shape[0]):
    X_trainNoise[i] = X_trainN[i] + 0.05 * np.random.randn(*X_trainN[i].shape)
	
## CONCANATE DATA:
X_train_Comb = np.concatenate((X_trainN, X_trainNoise), axis=0)
y_train_Comb = np.concatenate((y_train,  y_train), axis=0)
```

#### 2. To tackle this classification problem, I picked the LeNET structure that had been utilized for MNIST digit recognition. Following image illustrates how the LeNET looks like. 

![alt text][image4]

The final model architecture consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x24 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x24 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x32	 				|
| Fully connected		| outputs 400  									|
| Fully connected		| outputs 100  									|
| Fully connected		| outputs 43   									|
| Softmax				| etc.        									|
|						|												| 

Code block for the final network architecture that was summarized in the table above is shown below:
```python
def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables 
	# for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # SOLUTION: Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x24.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 24), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(24))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)

    # SOLUTION: Pooling. Input = 28x28x24. Output = 14x14x24.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x32.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 24, 32), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(32))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)

    # SOLUTION: Pooling. Input = 10x10x32. Output = 5x5x32.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x16. Output = 800.
    fc0   = flatten(conv2)
    
    # SOLUTION: Layer 3: Fully Connected. Input = 800. Output = 400.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(800, 400), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(400))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # SOLUTION: Activation.
    fc1    = tf.nn.relu(fc1)

    # SOLUTION: Layer 4: Fully Connected. Input = 400. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(400, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # SOLUTION: Activation.
    fc2    = tf.nn.relu(fc2)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 43.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits
```

#### 3. The model is trained for 15 epochs to achieve 94% validation accuracy. Adam optimizer is used with learning rate of 0.0025. The batch size of 128 worked well.

.
   
#### 4. For finding a solution and getting the validation set accuracy to be at least 0.93, I utilized the LeNET architecture and changed some of its settings. LeNET seemed to be sufficient architecture to tackle this problem due to its similar success with the MNIST datasets.

My final model results were:
* validation set accuracy of 94%
* test set accuracy of 93%

I kept the 3 color channels of the input images and played with the fully-connected layers as I monitored the validation and test accuracies. 

My observation was that the data augmentation has to be done properly. Blindly applying any kind of transformation will hinder the training.
Data normalization helps as well as adding noise to the input images provides robustness. I did not see any need for dropout layers since did not 
observe much overfitting.


### Test a Model on New Images

#### 1. I downloaded 14 new images from the web to test the trained network on German traffic signs. Below are these normalized images.

![alt text][image5] 

The first image might be difficult to classify because they are randomly picked from the web which may lead to different values for the color channels.
Also, the shading and the location of the signs within the 32x32 pixels might be not known in the training set. While normalization help, model may not
generalize for these randomly picked and scaled images.


#### 2. Model prediction for these signs and how certain it is shown below with the images. Accuracy of 85% is achieved on these new 14 images. 

It appears to be that the "traffic light" sign and 'crosswalk' signs are mistaken as "general caution". 

#### 3. Below, I plotted the softmax probabilities as bar charts for each of the images and provided the top 5 softmax probabilities for each image along with the sign type of each probability. 

The code for making predictions on my final model is located towards the end of the Ipython notebook.

For the all images, the model seems to be very certain on the prediction even for the wrong prediction as shown for each of the images below.

![alt text][image6]  ![alt text][image7] 
![alt text][image8]  ![alt text][image9] 
![alt text][image10]  ![alt text][image11] 
![alt text][image12]  ![alt text][image13] 
![alt text][image14]  ![alt text][image15]
![alt text][image16]  ![alt text][image17]
![alt text][image18]  ![alt text][image19]
  
### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


