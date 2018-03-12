# Project 5 - Vehicle Detection Project -- 3rd Submission

### Zikri Bayraktar, Ph.D
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./images/car_notcar.png
[image2]: ./images/binned.png
[image3]: ./images/color_hist.png
[image4]: ./images/hog.png
[image5]: ./images/sliding_window.png
[image6]: ./images/test1_result.png
[image7]: ./images/test2_result.png
[image8]: ./images/test3_result.png
[image9]: ./images/test4_result.png
[image10]: ./images/test5_result.png
[image11]: ./images/test6_result.png
[image12]: ./images/consensus.png
[image13]: ./images/Windows.png
[video1]: ./images/ztest.mp4
[video2]: ./images/zproject.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  
In certain parts, I utilized the code provided in lectures and quizzes to complete the project.

---
### 1. Writeup / README
This document is the writeup on the project, and I will try to describe the steps I took to complete the project. Here is project
code in a [Jupyter Notebook](./P5_Combined_Pipeline.ipynb).

I approached to this projects in two steps. 

Step 1. Train classifiers to identify cars vs non-cars in images.

Step 2. Create a pipeline that can take an image from a video and identify the cars in that image by drawing a box around it.

### STEP 1. CLASSIFIER TRAINING

During training, I realized that most of my trained models easily achieved 99.X % accuracy on the testing data sets without significant tuning effort.
To achieve a better classification, I proceeded with generating multiple models to create an <b>ensemble</b> and utilize their
consensus on making the final decision.  Hence, I trained 5 different models.

These 5 models are based on the color spaces, i.e. 'RGB', 'HSV', 'HLS', 'YCrCb' and 'YUV'.  I also tried 'LUV' but some numeric issues came up
and I could not use it. 

In each model, I used 'spatial binning', 'color histogram' and 'hog features' combined for each of the images to classify.

The labeled data set was provided by Udacity for vehicle and non-vehicle examples to train the classifiers. 
These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), 
the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.

Below are couple of examples of `vehicles` on the left, and `non-vehicle` images on the right. These were all 64x64px PNG images that I used in training classifiers.
![alt text][image1] 

I utilized the linear support vector machine classifier `LinearSVC` from the `scikit-learn` package to train all 5 models.

Here are the jupyter notebook for each of the training model: [RGB](./Vehicle Classification RGB_hist_range.ipynb), 
[HSV](./Vehicle Classification HSV_hist_range.ipynb),
[HLS](./Vehicle Classification HLS_hist_range.ipynb),
[YCrCb](./Vehicle Classification YCrCb_hist_range.ipynb),
[YUV](./Vehicle Classification YUV_hist_range.ipynb).

| MODEL    | Training Accuracy  | Testing Accuracy  |
| ---------|:------------------:| -----------------:|
| RGB      | 1.0                | 0.9960            |
| HSV      | 1.0                | 0.9931            |
| HLS      | 1.0                | 0.9960            |
| YCrCb    | 1.0                | 0.9996            |
| YUV      | 1.0                | 0.9975            |


#### SPATIAL BINNING

Each training image is converted to a feature vector containing `spatial binning`, `color histogram` and `hog features`.
Spatial binning is achieved by the `bin_spatial` function and images are resized to 32px or 16px depending on the trained model.
Size reduction is found out empirically during the training and ensured consistency at inference.

Below is a car image 64x64 on the left and binned to 32x32px on the right. Many of the structural features are preserved even at a much smaller size.

![alt text][image2] 

#### COLOR HISTOGRAM

Another set of features are obtained through the color histogram of the images. Each image has 3 channels and the each channel binned into 
32 color bins and all channels concatenated to create the feature vector in the `color_hist` function. For each model with different color space, image is first converted to that color space
via `cv2.cvtColor` function and then histogram is applied.

Below is a sample image and the corresponding 'RGB' histogram. (shown as not normalized)

![alt text][image3] 

### Histogram of Oriented Gradients (HOG)

The last set of features are extracted from each image via HOG processing. For each model with different color space, I explored 
the extraction of the HOG features using `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=16`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`:

![alt text][image4] 

Through experimentation, I concluded on different values for each of the model.  `get_hog_features` is utilized to extract the HOG features per channel.

Once all features are concatenated, I utilized 'sklearn.StandardScaler' function to normalize the features before using in the training.

Trained models and scaler values are stored in Pickle files to be loaded and used at the inference step.


### Sliding Window Search in 1st Submission

Per recommendation from lecture, I focused on the lower portion of the image where we would expect the cars to be. I utilized the lecture functions, 
`slide window` for this task.  The y-pixel range of [400, 690] is divided into 64, and 128 pixel regions. I tried even smaller window sizes but it took significantly longer
to compute them all. Below images illustrates the region where the cars are searched.

![alt text][image5]

### Sliding Window Search in 2nd Submission

For the cars far away, I realized that I do not need to search the near grids.  In the second submission, I constructed 4 different regions to look
for cars with different windows sizes. The y-start-stop regions are as follows `y_ss = [[400, 496], [428, 556], [450, 642], [400, 668]]` with 
corresponding window sizes `wws = [48,64,96,128]`.  This way, I could search smaller windows sizes for detection of the white car when it is far away
more efficiently. This improved the detection accuracy significantly when combined with 0.75 overlap rate.  Following image illustrates the `y_start_stop` 
ranges for each search zone.

![alt text][image13]

### Sliding Window Search in 3rd Submission

Similar to 2nd submission, I utilized multiple search zones for 3 different window sizes. This time I only had windows sizes of `wws = [48,64,96]` and 
y-start-stop ranges of `y_ss = [[400, 496], [428, 600], [400, 642]]` with an overlap rate of 0.7.  


### Test Images

To demonstrate how the pipeline is working, I tested on the 6 test images provided. One critical thing that came out was that
utilizing the essembly of 5 models is hindering the performance, especially 'RGB' and 'HSV' models. 

At the end, I only let 3 models of the ensemble to come up with a consensus on deciding which pixels contain car images, and these are 
'YCrCb', 'HLS' and 'YUV' models.

![alt text][image6] 
![alt text][image7]
![alt text][image8] 
![alt text][image9]
![alt text][image10]
![alt text][image11]
 
---

### STEP 2. Video Implementation

Here is the results for [test video](./images/ztest.mp4) and [project video](./images/zproject.mp4). 

The video pipeline loads each frame as an image and applies the following steps. For each model in the ensemble, 128px and 64 px
sliding windows are computed and for each model, hot pixels are identified. A heatmap created but this is converted to a binary image,
which results in identified pixels per model. This binary image is added to the `consensus` of 3 models. 

Once, all models complete classification, `consensus` is thresholded by value of 2, i.e. 2 out of 3 models says there is a car, then
that is idenfied as car. This `consensus` then passed to `scipy.ndimage.measurements.label()` to identify individual blobs in the image.
I constructed bounding boxes to cover the area of each blob detected by `draw_labeled_bboxes`.  

To visualize, we can look at the image below. We see each model (YCrCb, HLS, YUV) provide a prediction where a car might be.
This information provided by the ensemble of models is combined into a single consensus and rectangles identifying cars are drawn on the image.

![alt text][image12]

### 2nd Submission Video Implementation:

To track the cars over multiple frames I used the `from collections import deque` function. This function helped to save heatmaps detected
over `n_frames` at a time. I tried couple of different values for the number of frames and 5 to work best.  One thing that I had to change
was that the `consensus` to reduce to 2 models removing the contributions from `HLS` leaving only `YCrCb` and `YUV` models to decide on the
pixels of the heatmap. After utilizing the `deque`, tracking of the cars became significantly more consistent.

`n_frames = 5`

`consensus = deque(maxlen = n_frames) `

### 3rd Submission Video Implementation:

In this submission, I utilized a secondary `deque` to keep track of the sum of the previous frames for `n_frames=4`. This allowed me to filter out the false
detections much easier, when coupled with the finer tuned sliding window schema. Final video demonstrates almost no false identification 
while continuously tracking both cars.


---

### Discussion

1. One of the main problems that I encountered was the handling of the PNG/JPEG images in OpenCV.  The training images were in PNG, which were
in range of (0,1). This caused me great deal of confusion, when I moved to inference step, where images were JPEG, and processed in range of (0,255).
Once, I realized where the differences came in, I fixed my models and predictions but spent quite a bit of time to make it consistent.

2. My initial decision to proceed with an ensemble of models led to very slow executing pipeline. For all models in the ensemble, I compute
a prediction and because each model is in a different color space, all features (including HOG) are computed multiple times. This slows down the
processing quite a bit (approx 7 sec per frame).  One improvement could be to eliminate this approach and go with a single model.

3. The pipeline seem to fail when the car passes under trees where the road is shaded. I presume this can be corrected by adding more images from
shaded region to the training data set. On the other hand, the pipeline seems to handle cars coming on the opposite lanes on the left handside, which
I was not expecting to handle.

4. As I completed this project, I find out there are many new ways of completing this project through deep learning. YOLO (You-Only-Look-Once), 
Single-Shot-Multiple-Detector, Fast RCNN etc. are some of the new methods that I am planning to implement to achieve object detection and tracking.
I believe, replacing these models with one of these models will speed up and provide better robustness.
  
5. In the second submission, for tracking cars over multiple frames, I used `deque` function from `collections` and it helped significantly.  I also
looked at the 3 model consensus and realized that `HLS` introduces significant artifacts to the heatmap beyond finding the car locations. To reduce
the false detections, I reduced the consensus only to 2 models.

6. In the second submission, I also changed where I searched for cars with varying windows sizes.  This was recommended in the lectures, but I 
had not implemented in the first submission.  Doing so, I detection accuracy increased and I was able to search using 4 different window sizes.

7. In the third submission, I spent quite sometime fine tuning the sliding windows, and combining it with a secondary `deque` to keep track of the 
thresholded frame sums to denoise the false identification. I simply exploited that neighboring frames do not change significantly if the car exists.
However, false identifications tend to come and go quite radically, so that second `deque` help filtering them out.
