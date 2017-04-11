import scipy
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from processing import process_image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import time

def video_image_processor_callback(img):
    result = process_image(img)
    return result

def process_images():
    for i in range(6):
        name = 'test_images/test{}.jpg'.format(i+1)
        inpimg = mpimg.imread(name)
        print("processing=",name)
        start = time.time()
        result = process_image(inpimg,key=name)
        stop = time.time()
        plt.title("{} {}m".format(name,(stop-start)/60))
        plt.imshow(result)
        plt.show()

def p(img):
    # Image call back for processing video
    result = process_image(img)
    return result

def process_video():
    video_ouptut = 'detection_video2.mp4'
    clip1 = VideoFileClip("test_video.mp4")
    #video_ouptut = 'project_detection_video.mp4'
    #clip1 = VideoFileClip("project_video.mp4")

    processed_clip = clip1.fl_image(p)
    processed_clip.write_videofile(video_ouptut, audio = False)

def process_frames():
    clip1 = VideoFileClip("test_video.mp4")
    #video_ouptut = 'project_detection_video.mp4'
    #clip1 = VideoFileClip("project_video.mp4")

    counter = 0
    for f in clip1.iter_frames():
        if counter == 6:
            break;

        mpimg.imsave("frames/frame{}.jpg".format(counter),f)
        processed_clip, heatmap,labels, total_labels, total_image,clean2 = process_image(f,save_heatmap=True,save_labels=True)
        mpimg.imsave("frames/processed{}.jpg".format(counter),processed_clip)
        mpimg.imsave( "frames/heatmap{}.jpg".format(counter),heatmap)
        print("labels-",labels)
        mpimg.imsave( "frames/labels{}.jpg".format(counter),labels[0])
        # mpimg.imsave( "frames/total_labels{}.jpg".format(counter),total_labels)
        mpimg.imsave( "frames/total_image{}.jpg".format(counter),total_image)
        mpimg.imsave( "frames/clean2-{}.jpg".format(counter),clean2)
        counter += 1

# process_images()
# process_video()
process_frames()
    
