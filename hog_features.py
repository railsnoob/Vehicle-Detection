from skimage.feature import hog
import cv2

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        # Use skimage.hog() to get both features and a visualization
        features, hog_image = hog(img, orientations=orient, 
                                    pixels_per_cell = (pix_per_cell, pix_per_cell),
                                    cells_per_block = (cell_per_block,cell_per_block),
                                    visualise = vis, feature_vector=feature_vec)
     
        return features, hog_image
    else:      
        # Use skimage.hog() to get features only
        features = hog(img, orientations=orient,
                    pixels_per_cell = (pix_per_cell, pix_per_cell),
                    cells_per_block = (cell_per_block, cell_per_block),
                    visualise = vis, feature_vector = feature_vec)
        return features
    



def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)


