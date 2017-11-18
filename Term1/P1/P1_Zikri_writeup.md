# **Finding Lane Lines on the Road** 

### Zikri Bayraktar

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road, and fit a straight line to annotate them.
* Reflect on your work in a written report describing the challenges addressed and any known shortcomings.


[//]: # (Image References)

[image1]: ./test_images_output/gray_image_solidWhiteCurve.jpg "Grayscale"
[image2]: ./test_images_output/canny_image_solidWhiteCurve.jpg "CannyEdgeDetected"
[image3]: ./test_images_output/masked_image_solidWhiteCurve.jpg
[image4]: ./test_images_output/with_lines_solidWhiteCurve.jpg
[image5]: ./test_images_output/solidWhiteCurve.jpg
   
---

### Code

The code for this project can be found [here](P1_Zikri.ipynb) 

### Reflection

### 1. Describe of the pipeline. 

My pipeline contains following steps:
> First, I converted the images to grayscale:

![grayscale_image][image1]

>> Then, applied Gaussian filter to the grayscale image and passed the image to Canny edge detection:

![canny_image][image2]

>>> Once, the edges detected, a mask is applied to filter out the regions beyond the Region of Interest (ROI). ROI is simply the area where we hope to find the lane lines.

![ROI_image][image3]

>>>> The masked image is passed to Hough transform to identify the coordinates of the lines. Using the coordinates, I computed the slope of each line and based on the slope, I applied filtering. Lines with negative slope belongs to left side of the masked image and slopes with positive sign belongs to right side. To eliminate horizontal lines, I filtered out slope magnitute less than 0.35 (i.e. <b>|slope|<0.35</b>). This removed the horizontal line coordinates providing more stable lane line detection.  Below image displays the lane lines in sections.

![Sectioned_lane_lines][image4]

>>>>> Next steps is a little algebra. First, I find the averaged coordinates of the left lines so that I can utilize this center point to fit a line that intersects the bottom of the image. Applied the same technique to the right lines so that the final image will have straight line fitted from the bottom of the image.  I also computed the furthest point of the lines, so that the line can be extended. Instead of extrapolating, I simply used the furthest point identified by the Hough transform. Once, all these computed, I combined the original image with the lane line prediction from the pipeline. I plot a green line on the left, and a blue line on the right.

![final_image][image5]


Finally, I applied the pipeline to the following two videos:

[Video1](test_videos_output/solidWhiteRight.mp4) <br>
[Video2](test_videos_output/solidYellowLeft.mp4) <br><br>

I also tackled the optional challenge video, which let to some of the methods I described above.<br>
[Challenge Video3](test_videos_output/challenge.mp4) <br><br>


### 2. Identify potential shortcomings with your current pipeline and suggest solutions

The optional challenge video actually showed some of shortcomings of my initial approach, which I incorporated int my code to handle these situations. Following shortcomings came out of the challenge video:

1. The video size was different then the initial test images and videos. I had to change the mask coordinates to variables instead of fixed values so that it can scale with the image dimensions of different video sizes. (I did not want to tackle this by re-sizing any input video to be the same. I wanted to keep the video in its original size and adjust the masking accordingly.) <br><br>

2. The challenge video includes a part of the car's hood at the bottom of each frames. This created horizontal edge detection, hence I implemented removal of lines with |slope|<0.35.  This removed the detection of the hood as well as some of the other horizontal lines identified on the road. <br><br>

3. The original images and videos have great contrast between lane lines and the road. However, the challenge video showed that the contrast may vary depending on the road type. Hence, in some frames, lane lines were not identified, which led to pipeline to crash due to empty arrays. I overcame this by simply not drawing any lines, if the Hough transform did not identified any coordinates. This is a simple work around. Real solution would be to carry over information from frame to frame. If we can estimate where the lane lines would be from previous frames, we can draw that line approximately. However, in its current form, pipeline does not allow information passing between two consequent frames. <br><br>

4. One short coming that I did not address is again due to the lack of contrast in the images, which can be due to shadows of the trees or simply type of the road. Asphalt roads are darker which creates good contrast with lane lines hence edges are detected very well. However, on concrete roads, lane lines cannot be distinguished properly. One solution could be to look at the contrast histrogram of each frame and come up with a auto-contrast adjusting method, or some contrast thresholding. That way, lane lines can be identified much better. (I assume similar issue could happend on snow; no contrast no lanes detected) <br><br>

5. The challenge video showed that the roads are not always straight, they usually bend. One of the short comings of the pipeline is that it tries to fit a straight line to a curved road. You can see from my solution to the challenge video that this creates a solution that is not stable. Lines keep jumping around. One solution could be that we can fit piece-wise linear lines to the curve, and even better solution would be actually fitting a polynomial curve. However, I did not address this problem. <br><br>

6. Another potential shortcoming is the presence of other cars. If a car enters into the Region of Interest (ROI) where we try to identify the lane lines, this can create many artificial edges which needs to handled properly so that they are not used in the computation of the lane lines. Similar issues with pot holes and road quality issues as well as problems with the old vs new lane drawings, where road crews may have drawn somewhat different lane lines than old lines. Any issue with contrast, will confuse this pipeline. <br><br>


