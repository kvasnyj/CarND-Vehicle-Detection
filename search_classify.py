import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from lesson_functions import *
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
#from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
import pickle
from moviepy.editor import VideoFileClip

# =============   HYPERPARAMETERS  =============
color_space = 'HLS'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 10  # HOG orientations
pix_per_cell = 16  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = 1 # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16)  # Spatial binning dimensions
hist_bins = 16  # Number of histogram bins
spatial_feat = False  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off
y_start_stop = [380, 650]  # Min and max in y to search in slide_window()
overlap = 0.5 # overlap fraction (common for x and y)
sizes_window = (64, 96, 128)
heat_threshold = 2
precompiled = 2 # 0 - no, 1 - features, 2 - features  and classifier
debug = False

# =============   PROCESS =============
svc = None
X_scaler = None

def process_image(image_src, debug = False):
    image = image_src.astype(np.float32) / 255

    windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                           sizes_window=sizes_window, overlap=overlap)

    # window_img = draw_boxes(draw_image, windows, color=(0, 0, 255), thick=2)
    if debug: print("total windows: ", len(windows))
    # plt.imshow(window_img)
    # plt.pause(0)

    hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space,
                                 spatial_size=spatial_size, hist_bins=hist_bins,
                                 orient=orient, pix_per_cell=pix_per_cell,
                                 cell_per_block=cell_per_block,
                                 hog_channel=hog_channel, spatial_feat=spatial_feat,
                                 hist_feat=hist_feat, hog_feat=hog_feat)
    if debug: print("hot windows: ", len(hot_windows))
    # window_img = draw_boxes(image, hot_windows, color=(0, 0, 255), thick=6)
    #plt.imshow(window_img)
    #plt.pause(0)

    heatmap = np.zeros_like(image[:, :, 0]).astype(np.float)
    heatmap = add_heat(heatmap, hot_windows)
    heatmap = apply_threshold(heatmap, heat_threshold)
    from scipy.ndimage.measurements import label
    labels = label(heatmap)
    labeled_windows = labeled_bboxes(labels)
    result = draw_boxes(image_src, labeled_windows, color=(0, 0, 255), thick=3)

    if debug:
        plt.imshow(result)
        plt.pause(0)

    return result

def prepare_classifier():
    cars = []
    notcars = []
    file = color_space + "_" + str(hog_channel) + "_" + str(hist_bins) + ".p"

    if precompiled == 0:
        for image in glob.glob('train_images/vehicles/**/*.png', recursive=True):
            cars.append(image)

        for image in glob.glob('train_images/non-vehicles/**/*.png', recursive=True):
            notcars.append(image)

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

        print('sample size (car, notcar):', len(car_features), len(notcar_features))

        X = np.vstack((car_features, notcar_features))
        X = X.astype(np.float64)
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        dist_pickle = {}
        dist_pickle["scaled_X"] = scaled_X
        dist_pickle["y"] = y
        dist_pickle["X_scaler"] = X_scaler

        pickle.dump( dist_pickle, open(file, "wb" ) )
    else:
        with open(file, "rb") as input_file:
            e = pickle.load(input_file)
            scaled_X = e["scaled_X"]
            y = e["y"]
            X_scaler = e["X_scaler"]
            if precompiled == 2: svc = e["svc"]

    if (precompiled==0)  or  (precompiled==1):
        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_X, y, test_size=0.2, random_state=rand_state)

        print('Using:', orient, 'orientations', pix_per_cell,
              'pixels per cell and', cell_per_block, 'cells per block')
        print('Feature vector length:', len(X_train[0]))

        svc = svm.SVC(kernel='rbf')
        #svc = LinearSVC()

        # Check the training time for the SVC
        t = time.time()
        svc.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

        dist_pickle = {}
        dist_pickle["scaled_X"] = scaled_X
        dist_pickle["y"] = y
        dist_pickle["X_scaler"] = X_scaler
        dist_pickle["svc"] = svc

        pickle.dump(dist_pickle, open(file, "wb"))

    return svc, X_scaler

# =============   Main code  =============

svc, X_scaler = prepare_classifier()

clip = VideoFileClip('project_video.mp4')#.subclip(39, 40)
if(not debug):
    new_clip = clip.fl_image(process_image)
    new_clip.write_videofile('project_video_result.mp4', audio=False)
else:
    image = mpimg.imread('test_images/test4.jpg')
    image = process_image(image, debug = True)