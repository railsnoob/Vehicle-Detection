# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

My submission for P5 Vehicle detection and tracking project for Udacity **Self Driving Car NanoDegree** . 

The goal of this project was to create a vehicle detection & tracking pipeline. The steps that were taken:

1. Performed a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images
2. Train a classifier Linear Support Vector Machine vehicle classifier using the massive GTA data
3. Apply a color transform and append binned color features, as well as histograms of color, to HOG feature vector to make it more robust
4. Implemented a sliding-window technique and used the trained classifier to search for vehicles in images.
5. Create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
6. Estimate a bounding box for vehicles detected in each frame of a video to be able to track the same.
7. We finish by listing out the problem with the methodology and what we can do to improve the system. 
