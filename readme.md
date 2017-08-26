**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction and color transform, as well as histograms of color, on a labeled training set of images and train a classifier Linear SVM classifier
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./images/car.png
[image2]: ./images/not_car.png
[image3]: ./images/windows.jpg
[image4]: ./images/heatmap_test6.jpg
[image5]: ./images/labels_test6.jpg
[image6]: ./images/output_test6.jpg
[image7]: ./images/test6.jpg


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]
![alt text][image2]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters including pix_per_cell and cell_per_block. I found the default setting works fine, it is hard to say which combination works best, as long as I used a largest number of features, so I just choose the default setting. 

However, I found the color space has more impact, where RGB seems to have worst performance. 

#### 3. Classifier training
I trained a linear SVM C = 100 and Gamma = 0.01, since I want better correctness of classification, on combination of hog features, color histogram and color space spatial features.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I am using 4 scales windows, each of them cover different area. I only search road area, whose pixel in y-axis is in [400, 656]. Since the size of closer car is bigger, smaller window covers a smaller area from 400 to 400 + max(size * 1.5, 656). 
![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?
Ultimately I searched on 4 scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  For this example image, 
![alt text][image7]

Here are some example images:
The heatmap created for test 
![alt text][image4]
After that, the heatmap is labeled and bounding boxes are drawn on that.

I tried different color spaces such as RGB, HSV and HLS, as well as YCrCb, where YCrCb provides best results, also, I tried RBF kernal and different combinations of (C, Gamma) values, where linear kernel with C = 100 and Gamma = 0.01 has good performance. 
### Here are six frames and their corresponding heatmaps:
![alt text][image5]
### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]
### Here the resulting bounding boxes are drawn onto the last frame in the series:


### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Then, I spent a lot of time to make the bounding boxes stable. Since I am working on a continuous frames of a video, I have a global heatmap M, recording previous heatmap. If a pixel is identified as a car in current frame, then I assume it should be car, if it is not a car in current frame but is a car at last frame, it will be preserved, assuming that classification at current frame is not perfect. But if a pixel is identified as not a car at several straight frames, I assume it is not a car. 
I achieve this by doing following, in each frame, the heatmap m after thresholding is precessed by assigning all non-zero element to be 10, and the global heatmap M will be averaged on m, i.e. M = (m + M) / 2, in this way, newly car pixel is 10, overlapped pixel between m and M is > 10, and old car pixel will be divided by 2.
The bounding box for the example image is

[Here](https://youtu.be/-jq4X-E60a0) is the result of detection for project video.


### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
We can see in the video that, there are some frames the classifier fails to recognize the car, the size and background of the car is pretty different from training data. I think the image after resizing is different with 64x64 training data, the interpolation used by resizing will miss some data points anyway. Therefore, I think data augmentation may be useful, training on some resized training image should be helpful for this.

Also, using multiple classifiers built in different color spaces is also helpful for this, but this has more computational complexity in test time.

