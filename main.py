import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage.measurements import label
import pickle
import time
    
ystart = 400
ystop = 656
scale = 1.5
spatial = 32


color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [400, None]


# Define a function to return some characteristics of the dataset 
def data_look(car_list, notcar_list):
    ''' 
        cars = list of name of cars images
        notcars = []
    '''
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(cars)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcars)
    # Read in a test image, either car or notcar
    img = np.array(cv2.imread(cars[0]))
    # Define a key "image_shape" and store the test image shape 3-tuple
    data_dict["image_shape"] = img.shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = img.dtype
    # Return data_dict
    return data_dict

# spatial = 32
# histbin = 32
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features 
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# def color_hist(img, nbins=32):    #bins_range=(0, 256)
#     # Compute the histogram of the color channels separately
#     channel1_hist = np.histogram(img[:,:,0], bins=nbins)
#     channel2_hist = np.histogram(img[:,:,1], bins=nbins)
#     channel3_hist = np.histogram(img[:,:,2], bins=nbins)
#     # Concatenate the histograms into a single feature vector
#     hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
#     # Return the individual histograms, bin_centers and feature vector
#     return hist_features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
ttt = False
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    global ttt
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
                # if ttt == False:
                #     print(color_space)
                #     ttt = True
        else: feature_image = np.copy(image)      

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
            # print('spatial_ ', len(spatial_features))
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
            # print('hist_ ', len(hist_features))
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
            # print('hog_ ', len(hog_features))
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features
    
# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

# def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
#                         hist_bins=32, orient=9, 
#                         pix_per_cell=8, cell_per_block=2, hog_channel=0,
#                         spatial_feat=True, hist_feat=True, hog_feat=True):
C = 100
gamma = 0.01
classifier_name = 'svc_{}_{}_{}.pickle'.format(str(C), str(gamma), color_space)

# train a SVM classifier
def train_classifier(cars, notcars, spatial, hist_bins):
    t = time.time()
    car_features = extract_features(cars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(notcars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
    print('compute features takes %d seconds' % (time.time() - t))

    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)) 
                     
    # Fit a per-column scaler
    tmp = np.array(X)
    print('XXX ', tmp.shape)
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.1, random_state=rand_state)

    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC 
    # svc = svm.SVC(C = C, gamma = gamma)
    svc = LinearSVC(C = 100, gamma = gamma)
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
    a = {}
    a['svc'] = svc
    a['X_scaler'] = X_scaler
    # save the classifier for future usage
    with open(classifier_name, 'wb') as handle:
        pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Saved SVM classifier as {}'.format(classifier_name))

def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    global ttt
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]
    # convert to desire color space
    ctrans_tosearch = np.copy(img_tosearch)  
    if color_space != 'RGB':
        if color_space == 'HSV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
    # scale the image is necessary
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    bboxes_list = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            # print(yb, xb)
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 

            if hog_channel == 'ALL':
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            else:
                hog_features = hog_feat1
            
            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)
            # print('spatial ', len(spatial_features), 'hist ', len(hist_features), 'hog', len(hog_features))

            # Scale features and make a prediction
            stacked = np.hstack((spatial_features, hist_features, hog_features))
            ss = np.array(stacked)
            # print('ssss ', ss.shape)
            test_features = X_scaler.transform(stacked).reshape(1, -1)
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                box = [[xbox_left, ytop_draw+ystart], [xbox_left+win_draw, ytop_draw+win_draw+ystart]]
                bboxes_list.append(box)

    return draw_img, bboxes_list

''' visualize dataset '''

# train a SVM classifier on training data, all training images are of fixed size of 64x64
def train():
    images_novehicle = glob.glob('./non-vehicles/*/*.png')
    images_vehicle = glob.glob('./vehicles/*/*.png')

    print(len(images_novehicle))
    print(len(images_vehicle))
    cars = []
    notcars = []
    for image in images_novehicle:
            notcars.append(image)

    for image in images_vehicle:
            cars.append(image)

    # load data
    data_info = data_look(cars, notcars) 

    print('Your function returned a count of', 
          data_info["n_cars"], ' cars and', 
          data_info["n_notcars"], ' non-cars')
    print('of size: ',data_info["image_shape"], ' and data type:', 
          data_info["data_type"])


    cur_time = time.time()
    train_classifier(cars, notcars, spatial, hist_bins)
    cur_time = time.time() - cur_time
    print('training costs %f second' % cur_time)


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes

    for box in bbox_list:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

# if a pixel is car on last frame and not a cat in current frame, 
First_time = False
count = 1
HeatM = None
def find_cars_in_image(img):
    blists = []
    global HeatM
    global First_time
    global count
    # for each scale of window, do window search and classification, get a list of bounding boxes.
    for i in range(len(scales)):
        ystop = ystops[i]
        scale = scales[i]
        out_img, box_list = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        blists.append(box_list)
        # plt.imshow(out_img)
        # plt.show()

    # create heatmap according to bounding boxes and apply threshold
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    for l in blists:
        heat = add_heat(heat,l)

    heat = apply_threshold(heat,1)
    plt.imshow(heat)
    plt.savefig('heatmap_test6.jpg')

    # in order to make windows between adjacent frame relatively stable, assign all nonzero element in heatmap to 10, 
    not_zero = np.nonzero(heat)
    xx = np.array(not_zero[0])
    yy = np.array(not_zero[1])
    heat[xx,yy] = 10

    if First_time == False:
        HeatM = heat
        First_time = True
    else:
        # average all element in previous heatmap and current heatmap, the car pixel in current frame will >= 5, pixel is a car before but not 
        # car currently will be divided by 2. Then, if a pixel is not a car in several straight frames, it will be discarded after next thresh
        HeatM = (HeatM * 0.5 + heat * 0.5)

    # as stated above, discard there pixels are not a car in straight frames
    HeatM = apply_threshold(HeatM, 1)
    feature_index = np.nonzero(HeatM)
    xx = np.array(feature_index[0])
    yy = np.array(feature_index[1])
    
    heatmap = np.clip(HeatM, 0, 255)
    labels = label(heatmap)
    plt.imshow(labels[0])
    plt.savefig('labels_test6.jpg')
    draw_img = draw_labeled_bboxes(np.copy(out_img), labels)
    count += 1
    if count > -1:
        plt.imshow(draw_img)
        plt.savefig('output_test6.jpg')
        plt.show()
    return draw_img

import os
import imageio
imageio.plugins.ffmpeg.download()
from moviepy.editor import VideoFileClip
from IPython.display import HTML
video_path = ""
def find_in_video(video_name):
    """
    Process video using VideoFileClip, apply find_line_in_image for each frame in video and create video of results
    Input: name of directory
    Output: None
    """
    input_video = video_name
    output_video = "output" + video_name
    white_output = os.path.join(video_path, output_video)
    white_input = os.path.join(video_path, input_video)
    clip1 = VideoFileClip(white_input)
    white_clip = clip1.fl_image(find_cars_in_image) 
    white_clip.write_videofile(white_output, audio=False)
    HTML("""
    <video width="960" height="540" controls>
      <source src="{0}">
    </video>
    """.format(white_output))


if __name__ == '__main__':
    with open('svc_100_0.01_YCrCb.pickle', 'rb') as handle:
        b = pickle.load(handle)

    svc = b['svc']
    X_scaler = b['X_scaler']
    Window_size = 64
    scales = [1, 1.25, 1.75, 2]
    ystops = [min(400 + int(i * 96),656) for i in scales]
    img = mpimg.imread('./images/test6.jpg')
    find_cars_in_image(img)
    # find_in_video('project_video.mp4')

    # draw sliding window demo
    # boxx = [[(0,400), (0 + int(Window_size * scales[0]), 400 + int(Window_size * scales[0]))]]
    # for i in range(1,4):
    #     x1 = boxx[-1][0][0] + int(Window_size * scales[i-1]) + 20
    #     print(boxx[-1][0][1], x1)
    #     b1 = (x1, 400)
    #     b2 = (x1 + int(Window_size * scales[i]), 400 + int(Window_size * scales[i]))
    #     l = [b1, b2]
    #     boxx.append(l)
    # cv2.rectangle(img, (0,400), , (0, 0, 255), 6)
    # cv2.rectangle(img, (int(Window_size * scales[0]),400), (int(Window_size * scales[0]) + int(Window_size * scales[1]), 400 + int(Window_size * scales[1])), (0, 0, 255), 6)
    # cv2.rectangle(img, (0,400), (0 + int(Window_size * scales[2]), 400 + int(Window_size * scales[2])), (0, 0, 255), 6)
    # cv2.rectangle(img, (0,400), (0 + int(Window_size * scales[3]), 400 + int(Window_size * scales[3])), (0, 0, 255), 6)
    # img = draw_boxes(img, boxx)

    # plt.imshow(img)
    # plt.savefig('windows.jpg')
    # plt.show()


