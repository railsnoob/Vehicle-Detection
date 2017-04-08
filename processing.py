import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pickle
from scipy.ndimage.measurements import label

from hog_features import get_hog_features

from persist import load_parameters,  load_svc_scaler

def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)



def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))
                        
def color_hist(img, nbins=32,bins_range=(0,255)):    #bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def extract_features(img,orient,pix_per_cell,cell_per_block,spatial_size,hist_bins):

    ch1 = img[:,:,0]
    ch2 = img[:,:,1]
    ch3 = img[:,:,2]
    # Get hog features for each color
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block,  vis=False, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block,  vis=False, feature_vec=False)

    # print("HOG 1 {} 2 {} 3 {}   ".format(hog1.shape,hog2.shape,hog3.shape));    
    # print("RAVELD HOG 1 {} 2 {} 3 {}   ".format(hog1.ravel().shape,hog2.ravel().shape,hog3.ravel().shape)); 
    hog_features = np.hstack((hog1.ravel(),hog2.ravel(),hog3.ravel()))
    
    # print(" hog_features {}".format(hog_features))
    
    # get the spatial fetures
    spatial_features = bin_spatial(img,size= spatial_size)
    hist_features = color_hist(img, nbins=hist_bins)

    # print(" spatial={} hist={} hog={}\n".format(spatial_features.shape,hist_features.shape,hog_features.shape))
    
    x = np.concatenate((spatial_features.ravel(), hist_features.ravel(), hog_features.ravel()))
    return x 
   
    
    return test_features
    # Get the
    
# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1
    cells_per_step = 3  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    print("nblocks_per_window:",nblocks_per_window)
        
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)

    bbox = np.empty((1,2,2))

    print("hog1",hog1.shape,"hog2",hog2.shape,"hog3",hog3.shape)
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            print("ypos",ypos,"xpos",xpos)
            print("hog_feat",nblocks_per_window, nblocks_per_window)
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            print("hog_feat1=",hog_feat1.shape,"hog_feat2=",hog_feat2.shape," hog_feat3=",hog_feat3.shape)
            
            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch1
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
            print("subimg shape=",subimg.shape)
            # Get color features
            spatial_features = bin_spatial( subimg, size=spatial_size )
            hist_features = color_hist( subimg, nbins=hist_bins )
            print("find_cars spatial={} hist={} hog={}\n".format(spatial_features.shape,hist_features.shape,hog_features.shape))
            # Scale features and make a prediction
            x = np.concatenate((spatial_features.ravel(), hist_features.ravel(), hog_features.ravel()))
            print("find_cars x=",x.shape)
            
            test_features = X_scaler.transform(x).reshape(1, -1)
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6)
                v = np.array([[ int(xbox_left), int(ytop_draw+ystart)],[int(xbox_left+win_draw),int(ytop_draw+win_draw+ystart)]]).reshape((1,2,2))
                print("V shape",v.shape)
                print("box",bbox.shape)
                bbox = np.vstack((bbox,v))

    bbox = np.delete(bbox,(0),0) # Remove the first element that I put on 
    print("bbox shape:",bbox.shape," values=",bbox)
    return draw_img,bbox




# Read in a pickle file with bboxes saved
# Each item in the "all_bboxes" list will contain a 
# list of boxes for one of the images shown above
# box_list = pickle.load( open( "bbox_pickle.p", "rb" ))

# Read in image similar to one shown above 
# image = mpimg.imread('test_image.jpg')


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    print(bbox_list);
    print("aaaa");
    for box in bbox_list:
        print("add_heat BOX", box)
        print(box[0][1],box[1][1])
        print(box[0][0],box[1][0])
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[int(box[0][1]):int(box[1][1]), int(box[0][0]):int(box[1][0])] += 1

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


def plot(orig, heatmap):
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(draw_img)
    plt.title('Car Positions')
    plt.subplot(122)
    plt.imshow(heatmap, cmap='hot')
    plt.title('Heat Map')
    fig.tight_layout()

    
def process_image(orig_img):

    print("Orig Image Shape: ",orig_img.shape)
    # Load svc and X_scaler from processed data
    
    svc, X_scaler = load_svc_scaler()

    scale =1
    # Load parameters
    orient, pix_per_cell, cell_per_block, spatial_size, hist_bins = load_parameters()

    ystart=456
    ystop =683
    
    img, bboxes = find_cars(orig_img,ystart,ystop,scale,svc, X_scaler,  orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat,bboxes)
    
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,1)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)

    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    
    return draw_img
