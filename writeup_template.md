**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/image0714.png
[image2]: ./output_images/car_hog.jpg
[image3]: ./output_images/extra1977.png
[image4]: ./output_images/noncar_hog.jpg
[image5]: ./output_images/search_window1.jpg
[image6]: ./output_images/testimg_window1.jpg
[image7]: ./output_images/testimg_window2.jpg
[image8]: ./output_images/heat_window1.jpg
[image9]: ./output_images/fitlered_heat_window1.jpg
[image10]: ./output_images/testimg1.jpg
[image11]: ./test_images/test1.jpg
[video1]: ./deep_processed.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

This is the write up for this project.

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code to develop the HOG feature extraction as well a spatial bin and color histogram is in the IPthyon Notebook 'dev_feature_extraction.ipynb'. The first cell block uses the functions that were created in Vehicle Detection lessons. The HOG extraction function is called 'get_hog_features' on line 39. It uses scitkit learn's hog function to extract HOG features from images.

I then copied the finalized functions to the python scripit 'prod_feature_extraction.py'. The one thing I changed in the 'extract_features' function in this file is added a color channel parameter for the HOG features since I found that HSV was one the best color spaces for car vs non-car features.

### Here is  car image and its HOG

![alt text][image1]
![alt text][image2]

### Here is a non-car and its HOG

![alt text][image3]
![alt text][image4]

####2. Explain how you settled on your final choice of HOG parameters.

In the Ipython Notebook 'dev_feature_extraction.ipynb' I tested several combinations of color channels and parameters. The code for this is in cell block 11-13. I used a joint scatter plot using the Seaborn library to display each color channel against each other. The joint plot also showed the histogram of values for each channel. YUV channel was a the other color channel I considered using for the HOG features but I found HSV using the parameters in the lesson gave the difference between car and non-car examples. I tried several combinations from the HOG parameters in the lesson but ultimately I found that the original parameters from class gave best distinct features between car and non-car examples.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The file I used to train a classifier is 'dev_vehicle_classifier.ipynb'. In cell block 1 I load required libraries. In cell block 2 the car images and non-car images were loaded, features extracted, scaled, shuffled and split into training at test sets. In cell block 3 a LinearSVC model was trained. In cell block 4 I used scikit learns SelectFromModel to select the most relevant features from the features set. This reduced the feature vector to 2946 from 7080. IN cell block 5 I trained another LinearSVC on the reduced feature vector. The results were comparable to the full feature vector. I saved this LinearSVC classifier, scaler and feature selector to a pickle so it could be loaded for the sliding window search. I continued to test other classifiers between cell block 7 to 9. 

From cell block 10 to 14 I designed a keras deep network classifier. I still used the reduced feature vector to train the keras network. I saved this model to file as well to test a LinearSVC vs a deep network for video processing.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I completed this portion of the project using trial and error. This can be found in the Ipython Notebook 'dev_sliding_window.ipynb'. The final production code for the sliding window functions are in the python script 'prod_sliding_window.py'. I would search each of the provided test images using different pixel sizes and overlapping percentages to see what the results were and how it impacted run time. Once I saw the results of the detected windows I would reduce the search area down areas where that pixel scale made sense to search.

Once I had tested several window sizes I brought them all together to see what the detected windows looked like. Again I had to go back to use trial and error to reduce the number of search windows. I also created a few left hand search grids since I found the white vehicle on the left side of the image was not being detected. I had a feeling it had to do with the search grid that the grid stopped before it reached the left edge of the images/video frame. When I introduced the left search grids the white vehicle was detected more often by the classifier.

![alt text][image5]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I tried several methods to optimize my classifier. This can be found in IPython Notebook 'dev_vehicle_classifier.ipynb'. The one method I did not use was increasing the amount of training data which in the end would have probably yielded the best result. I use a SelectFromModel function in scikit learn to reudce the number of feature vectors. I tried several parameters to optimize my LinearSVC classifier but in the end I had too many false positives or the white vehicle was not being detected. I tried a decision tree but I still had the same result that eitehr the white vehicle was not being detected or if I increased the number of search widnwos there were too mach false potives being detected.

I decided to go back and try building a Keras model (as previously mentioned). I still used the same reduced feature vector for the Keras model. This classifier worked much better by reducing the amount of false positives and being able to detect the white vehicle. I found areas with trees had significant flase poistives with the LinearSVC and decision tree but the keras classifier had much smaller issues with these areas in the video.

![alt text][image6]
![alt text][image7]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./deep_processed.mp4). It is called deep_processed.mp4 if the link does not work.


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. The positive detections were then layered on a heatmap. Anything less than 1 layer was considered to be a false positive and was filtered out of the heatmap. THe filtered result was then used to draw a box around the labelled heatmap detections. The final video processing pipeline is Ipython Notebook 'prod_sliding_window.ipynb'.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here is a test image with its heatmap
![alt text][image11]
![alt text][image8]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap for the test image
![alt text][image9]

### Here the resulting bounding boxes are drawn onto the test image
![alt text][image10]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

This project was quite the learning opportunity for me. I tried several classifier parameters to reduce the amount of false positives and to try and detect the white vehicle. I went and really examined the car and non-car color spaces and features to reduce it down to what I found to be the best selection. Even with this I found my initial classifers had specific areas of the project video that it detected a lot of false positives and it often would not detect the white car. I did use the time to learn how to use the seaborn library to find useful features from different color spaces of the images. After I went back and tweaked the feature extraction I detected the black car nearly the entire video but the whtie car would still drop off detection and trees would often be detected as a car.

ANohter issue I ran into is my search windows. I believe the amount of search windows I have are really slowing down the processing of the images. I tried reducing the amount of widnwos but it imapcts the ability to detect cras in certain parts of the project video.

I found that most classifiers I trained had issues that I described above. This why I decided to try using a keras deep network classifier and I noticed an instant differnce, expect I had designed a network that was too connected and it would have taken 10 hours to process the 50 second video. Obviously that is not a production solution since it would need ot run in real time. I continued to play with the keras network until I created the final one in the IPython Notebook. It would run just as a fast as the LinearSVC solution I had created, it took an hour to process the video, but it detects the cars in the video nearly the entire time and has very little false positives. Again this is not a production solution since it took an hour to process 50 seconds of video. So this pipeline will fail if it had to detect cars in real time.

I also extract the HOG features from each indivudal image. To improve run time I could extract the HOG features once from the video fram and slect the features based on the current search widnow to help improve run time.

Thas pipeline is also likely to fail in treed areas since I think of the edges of the tree can closely match the HOG signature of a car. If more trees were included as non-car examples in training it would most likely reduce this issue.

This pipeline is also likely to fail if other vehicles were on the road that the classifier was not trained on. Again the easiest solution would be to add the different vehicle types to the training set.

This pipeline will also likely fail if the horizon were to change positions, such as going up or down hill. The search window grid is designed on a fixed horizon position. Since the horizon could change positions in the video because the car is going up or down hill then it could potential no longer detect a vehicle infront of it becuase it has moved out of teh search grid area. If the search grid area could change with the vehicles detected steepness this would reduce this issue.



