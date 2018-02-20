## Advanced Lane Line Detection Project
## Udacity Self Driving Car Engineer Nanodegree, Term 1, Project 4

#### Zikri Bayraktar, Ph.D.

---
I utilized codes from the lectures to complete this project throughout.

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./report_images/Project_figure_1.png "Undistorted"
[image2]: ./report_images/Project_figure_2.png "Binary Example"
[image3]: ./report_images/Project_figure_3.png "Masked Example"
[image4]: ./report_images/Project_image_4_1.png "Straight Perspective Example"
[image5]: ./report_images/Project_figure_4.png "Perspective Example"
[image6]: ./report_images/Project_4_2.png "Histogram"
[image7]: ./report_images/Project_figure_5.png "Lane Filled"
[image8]: ./report_images/Project_figure_6.png "Lane Filled Original"
[image9]: ./report_images/Figure_unwarped.png "Unwarped Image"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

In this document, I will explain each step on how I approached this project.
The code for this project can be found [Here](Zikri_T1_P4_Advanced_Lane_Lines_v05.ipynb)
as a Jupyter Notebook. 

### Camera Calibration

First step is to calibrate the camera per given checkboard images provided by Udacity by finding the corners using `cv2.findChessboardCorners`. 
This is a crucial initial step and we compute the camera matrix and distortion coefficients using the `cv2.calibrateCamera`.

The code for the camera calibration is under Section 1. I did not carry out this inside a function
so that the <b>mtx</b>, and <b>dst</b> variables will be available through the code as global variables.

Below is an image where the original camera image of a chessboard is undistorted and shown side-by-side.

![alt text][image1]

### Pipeline (single image example shown)

For the rest of the project, I tried to split the code into individual function that
carry out each task listed in the goals section above. The pipeline calls each function
per image and applies the steps.

To demonstrate these steps, I will use a sample image as shown (original image) below:

First step is off course applying camera correction coefficients. We can see the difference between the camera captured original image and 
the corrected image on the right. Applied correction is clearly visible on the hood of the car at the bottom of the right image. 
![alt text][image9]


Then, this image is passed to `threshold_gradient_and_color()` function for edge detection.
I applied sobel_x, and s- and l-channels of HLS channels. The image after this thresholding looks like this:

![alt text][image2]

We can see above in the binary image that cars in the next lane also come up. I used the same masking function
that I used in Project 1 to mask the region of image where our lane lines are in front of us.

This function is `region_of_interest()` and following image is the result:
![alt text][image3]


Then, next step is the perspective transform of the above image through `warper()` function. This function employes
`cv2.getPerspectiveTransform()` and `cv2.warpPerspective()` in to warp the binary image into a "birds-eye view".

However, we first need to make sure that we can correctly do this transform on straight lanes.  To test that
I utilized the following image and here are the straight lanes perspective transform:

![alt text][image4]

Once, we confirmed on the straight lane, we can transform our curved lanes as well as shown below:
![alt text][image5]

Next step is to utilize the sliding window approach from the Udacity lectures and locate the lane pixels so that we
can fit a second degree polynomial to each of the lane curves. All this is achieved in `sliding_window()` function.

Below is an histogram of the curves to demonstrate how we identify where the lanes are:
![alt text][image6]

The end result of this function looks like this with the curves identified and the lane region is filled:

![alt text][image7]

The function `sliding_window()` computes the fitted polynomial coefficients, and then the radius of the curvature for both left and right lanes.
It returns an average of these two curvatures as the predicted curvature.
It also computes the shift of the car from the center of the lane and all is displayed over the original image as shown below.
(Note that I used another image below due to fact that concrete paved surfaces are more challenging to detect):

![alt text][image8]



---

### Pipeline (video)

#### 1. Here is a link to the final video output with the lane shaded green. It works very well but I will discuss more below in the Discussion section on how to make it better.

Here's a [link to my video result](report_images/project_video_run7.mp4)

---

### Discussion

I implemented this project in rather a naive way that everything is computed for every single frame rather than recording
some of the values and remembering in consecutive frames. This can create issues if the pipeline fails to clearly calculate the 
curvatures or fails to extracts the lane lines. More robust approach would be to store the computed values and carry them over to
next frames. 

One clear example of such failures could be where the lane lines almost completely disappear on concrete roads. The white and yellow
lanes cannot be distinguished well on concrete. I extracted each frame from the test video and I focused on tuning my thresholds on concrete pavement frames of the video. This could be addressed
by storing previous frame values and utilize them sensibly in frames where we cannot identify the lane lines.
 
Another issue might be with the image size coming from the camera. If the camera were to be replaced with a wider view camera,
it has to be re-calibrated and some of the hard-coded values had to be changed.  I did not spend too much time on making it robust in this way.
 
I also read on the discussion board that this could have been done through Deep Learning but I did not approach it that way since all the instructions were
focusing on the computer vision approach.

