
# coding: utf-8

# # Feature Extraction Production
# 
# This file contains the production functions for vehicle feature extraction

# ## Define feature extraction functions

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import StandardScaler
import glob
import os
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog

def bin_spatial(img, size=(32, 32)):
    """
    This function takes as input: an image called 'img' and a tuple of (x,y) called 'size'.
    
    It returns 'img' scaled to dimensions x by y as a xy by 1 array.
    """
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features
 
def color_hist(img, nbins=32, bins_range=(0, 256)):
    """
    This function takes as input: a three channel image called 'img', a number of bins called 'nbins' and a range for the
    bins called 'bin_range'.
    
    It returns 3 histograms, one for each channel of 'img', as a numpy array.
    """
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    #hist_features = np.concatenate(( channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        features,hog_image=hog(img, orientations=orient,pixels_per_cell=(pix_per_cell,pix_per_cell),
        cells_per_block=(cell_per_block,cell_per_block), transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec)
        
        return features, hog_image
    else:      
        # Use skimage.hog() to get features only
        features=hog(img, orientations=orient,pixels_per_cell=(pix_per_cell,pix_per_cell),
        cells_per_block=(cell_per_block,cell_per_block), transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec)
        
        return features


def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True, hog_color_space="HSV"):
    """
    This function takes as input: a list of image paths called 'imgs', a color space to convert images to
    from the following list, ['RGB','HSV','LUV','HLS','YUV'], called 'cspace', a tuple of x and y values
    to scale images to called 'spatial_size', a number to bins for a histogram called 'hist_bins' and a tuple
    for the start and stop of a histogram called 'hist_range'.
    
    It returns a list of numpy arrays of bin features and color histogram features for each image in 
    'imgs'
    """
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
            if hog_color_space != 'RGB':
                if hog_color_space == 'HSV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                elif hog_color_space == 'LUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
                elif hog_color_space == 'HLS':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
                elif hog_color_space == 'YUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
                elif hog_color_space == 'YCrCb':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            else: feature_image = np.copy(image) 
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            elif hog_channel=='GRAY' :
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                hog_features = get_hog_features(gray, orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features