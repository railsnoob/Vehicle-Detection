import numpy as np
from hog_features import get_hog_features, convert_color
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def show_hog_features(imgname,title):
    img = mpimg.imread(imgname)
    print(img.shape)
    plt.title("Image")
    plt.imshow(img)
    plt.show()
    img = convert_color(img, conv= 'RGB2YCrCb')
    plt.title("Color Converted")
    plt.imshow(img)
    plt.show()
    print(img.shape)
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2

    ch1 = img[:,:,0]
    ch2 = img[:,:,1]
    ch3 = img[:,:,2]
    
    features1, hog_image1 = get_hog_features( ch1, orient, pix_per_cell, cell_per_block, vis=True)
    features2, hog_image2 = get_hog_features( ch2, orient, pix_per_cell, cell_per_block, vis=True)
    features3, hog_image3 = get_hog_features( ch3, orient, pix_per_cell, cell_per_block, vis=True)
    plt.title(title)
    plt.imshow(hog_image1)
    plt.show()
    plt.imshow(hog_image2)
    plt.show()
    plt.imshow(hog_image3)
    plt.show()

show_hog_features('d/vehicles/GTI_MiddleClose/image0186.png',"vehicle");
show_hog_features('d/non-vehicles/GTI/image1515.png',"non-vehicle");
