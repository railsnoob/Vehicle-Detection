import scipy
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from processing import process_image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def video_image_processor_callback(img):
    result = process_image(img)
    return result

def process_images():
    for i in range(6):
        inpimg = mpimg.imread('test_images/test{}.jpg'.format(i+1))
        result = process_image(inpimg)
        plt.title("Test Image {} - Final Result".format(i+1))
        plt.imshow(result)
        plt.show()

def p(img):
    # Image call back for processing video
    result = process_image(img)
    return result

def process_video():
    video_ouptut = 'detection_vide.mp4'
    clip1 = VideoFileClip("project_video.mp4")

    processed_clip = clip1.fl_image(p)
    processed_clip.write_videofile(video_ouptut, audio = False)

process_images()
    
