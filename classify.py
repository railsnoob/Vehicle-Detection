import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from hog_features import get_hog_features
from processing import  extract_features
import time

from persist import load_svc_scaler, save_svc_scaler, load_parameters, save_parameters, pickle_data, load_data

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler

# import svc
from sklearn.metrics import accuracy_score

from sklearn.svm import SVC


images = glob.glob('*.jpeg')
cars = []
notcars = []

# for image in images:
#    if 'image' in image or 'extra' in image:
#        notcars.append(image)


def draw_boxes(img, bboxes, color=(0,0,255), thick=6):
    draw_img = np.copy(img)
    
    for box in bboxes:
        cv2.rectangle(draw_img, box[0], box[1], color, thick)
    return draw_img

    
def test_code():
    ind = np.random.randint(0,len(cars))
    image = mpimg.imread(cars[ind])

    features, hog_image = get_hog_features(image, 9, 8,4,True,True)
    
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(image, cmap = 'gray' )
    plt.title('Example Car Image')
    
    plt.subplot(122)
    plot.imshow(hog_image, cmap = 'gray' )
    plot.title('HOG Visulization')


# img = mpimg.imread('test_image.jpg')
    
def read_data(limit=5):

    # X=np.empty((1,64,64,3)) This is for reading images.
    

      
    params = { "orient" : 9 ,
                "pix_per_cell" : 8,
                "cell_per_block" : 2,
                "spatial_size" : (32,32),
                "hist_bins" : 32
                }

    ## 32*3
    imgsz = 64
    ## Should really be imgsz/pix_per_cell - 
    total_length = params["hist_bins"]*3 + 3*params["spatial_size"][0]*params["spatial_size"][1] + ( (imgsz/params["pix_per_cell"]) - 1)*((imgsz/params["pix_per_cell"])-1)*params["orient"]*params["cell_per_block"]*params["cell_per_block"]*3
    
    print("Ttoal length", total_length)
    X = np.empty((int(total_length),))
    Y=np.empty([])
    counter = 0
    save_parameters(params);
    
    orient = params["orient"]
    pix_per_cell = params["pix_per_cell"]
    cell_per_block = params["cell_per_block"]
    spatial_size = params["spatial_size"]
    hist_bins = params["hist_bins"]

    vehicles = glob.glob("vehicles/*/*.*")
    non_vehicles = glob.glob("non-vehicles/*/*.*")

    if limit:
        vehicles = vehicles[:limit]
        non_vehicles = non_vehicles[:limit]

    for f in vehicles:
        xv = mpimg.imread(f)
        xv = extract_features(xv,orient,pix_per_cell,cell_per_block,spatial_size,hist_bins)
        yv = 1
        X = np.vstack((X,[xv]))
        Y = np.vstack((Y,[yv]))
        print("Counter Vehicles:",counter)
        counter += 1
        if (limit != None and counter >= limit):
            break
    counter = 0
    print("After vehicles ",X.shape,Y.shape,Y)
    
    for f in non_vehicles:
        xnv = mpimg.imread(f)
        xnv = extract_features(xnv,orient,pix_per_cell,cell_per_block,spatial_size,hist_bins)
        ynv = 0
        X = np.vstack((X,[xnv]))
        Y = np.vstack((Y,[ynv]))
        print("Counter Non Vehicles:",counter)
        counter += 1
        if (limit != None and counter >= limit):
            break

    print("Before DELETE NON vehicles ",X.shape,Y.shape)
    X = np.delete(X,(0),0)
    Y = np.delete(Y,(0),0)
    print("After NON vehicles ",X.shape,Y.shape)

    # plt.figure(figsize=(10,10))
    
    # for i in range(X.shape[0]):
    #     plt.subplot(X.shape[0],1,i+1)
    #     plt.imshow(X[i])

    # plt.show()
    
    
    # pickle_data(X,Y,"train")

    return X, Y

def explore_data(X_train,X_test,Y_train,Y_test):
    pass
    #

def get_params():
    # Try load params from pickle
    pass
        
def classify():

    params = { "orient" : 9 ,
               "pix_per_cell" : 8,
               "cell_per_block" : 2,
               "spatial_size" : (32,32),
               "hist_bins" : 32
               }
    save_parameters(params)

    # Load X and y
    data = load_data('processed_data')
    if (data == None):
        data = read_data(limit=None)

    seed = 43
    
    X,y = data

    print(" Xshape {}".format(X.shape))
    print(" yshape {}".format(y.shape))
    
    X_scaler = StandardScaler().fit(X)
    X = X_scaler.transform(X)
    
    # Split into train and test
    n_folds = 10
    
    X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size =0.2,random_state = seed)

    print("X_train",X_train.shape)
    print("y_train",Y_train.shape)
    print("X_test",X_test.shape)
    print("Y_test",Y_test.shape)

    Y_train = Y_train.reshape((Y_train.shape[0]))
    Y_test = Y_test.reshape((Y_test.shape[0]))
    
    pickle_data(X,y,"processed_data");

    kfold = KFold(n_splits= 10, random_state= seed)
    
    model = SVC()
    

    
    print("Started FIT",time.asctime())
    t=  time.time()
    model.fit(X_train, Y_train)
    t2 = time.time()
    print(round(t2-t,2),'Seconds to train SVC ...')

    print('Test Accuracy of SVC = ',round(model.score(X_test,Y_test),4))
    
    predictions = model.predict(X_test)


    #cv_results = cross_val_score(model,X_train, Y_train, cv=kfold,scoring='accuracy')
    #print("Results {}".format(cv_results))
    
    save_svc_scaler(model,X_scaler)
    
    print("Test set score ")
    print(accuracy_score(Y_test,predictions))
    print(confusion_matrix(Y_test,predictions))
    print(classification_report(Y_test,predictions))
    # Print Test Error
    # shuffle both

classify()


    
    

    

    
