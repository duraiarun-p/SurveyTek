import numpy as np
import math
# import os.path
from os import listdir
from os.path import isfile, join
import glob
# from imutils import paths
from cv2 import imshow, waitKey,  destroyAllWindows,filter2D, imread, resize
from theano.tensor.signal import pool
from theano import tensor as T
from theano import function
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#%%

def imageMaxPool(img):

    my_dmatrix = T.TensorType('uint8', (False,)*2)
    input = my_dmatrix('input')
    maxpool_shape = (30, 40)
    pool_out = pool.pool_2d(input, maxpool_shape, ignore_border=True)
    f = function([input],pool_out)

    #invals = numpy.random.RandomState(1).rand(3, 2, 5, 5)
    invals = img
    output = f(invals)
    return output


def showPicture(img):
    imshow('image',img)
    print("Showing Picture")
    waitKey(0)
    destroyAllWindows()


def centeredFilter(N):
    mask_shell = np.empty((N, N), dtype=np.object)
    num = N//2
    y = num
    x = -1 * num
    countX = 0
    countY = 0

    for j in mask_shell: # interate over rows
        for i in j: # interate over elements
            mask_shell[countX][countY] = [x,y]
            x = x + 1
            countY = countY + 1
        y = y - 1
        x = -1 * num
        countY = 0
        countX = countX + 1

    return mask_shell


def steerableGaussian(img, theta, kernel_size):
    G2A_array = np.empty((kernel_size, kernel_size))
    G2B_array = np.empty((kernel_size, kernel_size))
    G2C_array = np.empty((kernel_size, kernel_size))
    countX = 0
    countY = 0
    # Calculate G2A - in array not normalized
    for rows in centeredFilter(kernel_size):
        for XY in rows:
            G2A = 0.9213*(2*(XY[0]*XY[0])-1)*math.exp(-1*((XY[0]*XY[0])+(XY[1]*XY[1])))
            G2A_array[countX][countY] = G2A
            countY = countY + 1
        countY = 0
        countX = countX + 1


    G2A_array = (math.cos(theta)*math.cos(theta))*G2A_array

    R2A = filter2D(img, -1, G2A_array)

    # Reset index variables
    countX = 0
    countY = 0
    # Calculate G2B - in array not normalized
    for rows in centeredFilter(kernel_size):
        for XY in rows:
            G2B = 1.843*(XY[0]*XY[1])*math.exp(-1*((XY[0]*XY[0])+(XY[1]*XY[1])))
            G2B_array[countX][countY] = G2B
            countY = countY + 1
        countY = 0
        countX = countX + 1

    G2B_array = (-2*math.cos(theta)*math.sin(theta))*G2B_array

    R2B = filter2D(img, -1, G2B_array)

    # Reset index variables
    countX = 0
    countY = 0
    # Calculate G2A - in array not normalized
    for rows in centeredFilter(kernel_size):
        for XY in rows:
            G2C = 0.9213*(2*(XY[1]*XY[1])-1)*math.exp(-1*((XY[0]*XY[0])+(XY[1]*XY[1])))
            G2C_array[countX][countY] = G2C
            countY = countY + 1
        countY = 0
        countX = countX + 1

    G2C_array = (math.sin(theta)*math.sin(theta))*G2C_array
    R2C = filter2D(img, -1, G2C_array)
    addAB = np.add(R2A,R2B)
    F_theta = np.add(R2C,addAB)

    # showPicture(F_theta)

    return F_theta


def steerableHilbert(img,theta,kernel_size):

    H2A_array = np.empty((kernel_size,kernel_size))
    H2B_array = np.empty((kernel_size,kernel_size))
    H2C_array = np.empty((kernel_size,kernel_size))
    H2D_array = np.empty((kernel_size,kernel_size))

    countX = 0
    countY = 0

    for rows in centeredFilter(kernel_size):
        for XY in rows:
            H2A = 0.9780*((-2.254*XY[0])+(XY[0]*XY[0]*XY[0]))*math.exp(-1*((XY[0]*XY[0])+(XY[1]*XY[1])))
            H2A_array[countX][countY] = H2A
            countY = countY + 1
        countY = 0
        countX = countX + 1


    H2A_array = (math.cos(theta)*math.cos(theta)*math.cos(theta))*H2A_array
    R2A = filter2D(img, -1, H2A_array)
    # R2A_LC = (math.cos(theta)*math.cos(theta))*R2A

    #showPicture(R2A)

    # Reset index variables
    countX = 0
    countY = 0

    for rows in centeredFilter(kernel_size):
        for XY in rows:
            H2B = 0.9780*(-0.7515+(XY[0]*XY[0]))*XY[1]*math.exp(-1*((XY[0]*XY[0])+(XY[1]*XY[1])))
            H2B_array[countX][countY] = H2B
            countY = countY + 1
        countY = 0
        countX = countX + 1

    # print(H2B_array)

    H2B_array = (-3*math.cos(theta)*math.cos(theta)*math.sin(theta))*H2B_array

    R2B = filter2D(img, -1, H2B_array)

    #showPicture(R2B)

    # Reset index variables
    countX = 0
    countY = 0

    for rows in centeredFilter(kernel_size):
        for XY in rows:
            H2C = 0.9780*(-0.7515+(XY[1]*XY[1]))*XY[0]*math.exp(-1*((XY[0]*XY[0])+(XY[1]*XY[1])))
            H2C_array[countX][countY] = H2C
            countY = countY + 1
        countY = 0
        countX = countX + 1

    H2C_array = (3*math.cos(theta)*math.sin(theta)*math.sin(theta))*H2C_array

    R2C = filter2D(img, -1, H2C_array)
    # Reset index variables
    countX = 0
    countY = 0

    for rows in centeredFilter(kernel_size):
        for XY in rows:
            H2D = 0.9780*((-2.254*XY[1])+(XY[1]*XY[1]*XY[1]))*math.exp(-1*((XY[0]*XY[0])+(XY[1]*XY[1])))
            H2D_array[countX][countY] = H2D
            countY = countY + 1
        countY = 0
        countX = countX + 1

    H2D_array = (-1*math.sin(theta)*math.sin(theta)*math.sin(theta))*H2D_array
    R2D = filter2D(img, -1, H2D_array)
    addAB = np.add(R2A, R2B)
    addABC = np.add(R2C, addAB)
    F_theta = np.add(R2D, addABC)

    # showPicture(F_theta)

    return F_theta


if __name__ == "__main__":

    '''
    Go through sheffield dataset
    and calculate pooled data
    '''
    # Datadir = 'Dataset\\'
    # Datadir='/home/arun/Documents/PyWSPrecision/Defect_Classification-main/Dataset/'
    Datadir='/home/arun/Documents/PyWSPrecision/datasets/cat_surveytek_1/train/'
    
    lda_input = np.array([], dtype=np.uint8)

    # array of number of pictures for each class in order from 1 - 40

    folders=listdir(Datadir)
    # del folders[0]    
    
    class_size = [316, 200]
    # classes = [1, 2]
    classes = np.arange(len(folders))+1
    # class_names = ['Mould', 'WaterStain']
    class_names=folders
    # for fi in range(len(folders)): class_names[fi]=folders[fi]
    class_labels = np.array([], dtype=np.uint8)
    ext = ['png', 'jpg', 'gif']
    
    # iterate over all images
    for index in range(len(classes)):
        class_label = classes[index]
        imdir = Datadir + class_names[index]
        # for path in paths.list_images(imdir):
        # for path in os.listdir(imdir):
        # files = []
        # [files.extend(glob.glob(imdir + '*.' + e)) for e in ext]
        imdir1=imdir+'/'+'*.jpg'

        files=glob.glob(imdir1)
        
        for path in range(len(files)):
            #print(path)
            img = imread(files[path], 0)
            img = resize(img, (1200, 900))
            class_labels = np.append(class_labels, class_label)
            kernel_size = 5
            theta_list = [0, 45, 90, 135, 180, 225, 270, 315]
            feature_maps = []
            for theta in theta_list:
                feature_maps.append(steerableGaussian(img, theta, kernel_size))
                feature_maps.append(steerableHilbert(img, theta, kernel_size))
            ds_vector = np.array([], dtype=np.uint8)
            for feature_map in feature_maps:
                pooled = imageMaxPool(feature_map)
                ds_vector = np.append(ds_vector, pooled)
            #print('ds_vector shape = ', ds_vector.shape)

            # ds_vector 14400 dimensional vector for single image
            if np.any(lda_input):
                try:
                    lda_input = np.stack((lda_input, ds_vector), axis=0)
                except:
                    ds_vector = np.resize(ds_vector, (1, 14400))
                    lda_input = np.concatenate((lda_input, ds_vector), axis=0)
            else:
                lda_input = ds_vector

            #print(lda_input.shape)

    '''
    dimensionality reduction by linear discriminant analysis from 14400 to n_components = numclass-1 = 1
    The value of n_components can be in range of 1 to numclass-1
    '''
    np.save("defect_dataset_14400_2", lda_input)

    lda = LinearDiscriminantAnalysis(n_components=1)
    dataset = lda.fit(lda_input, class_labels).transform(lda_input)

    #print(dataset)
    #print(dataset.shape)
    #print(class_labels)

    np.save("defect_class_labels_2", class_labels)
    np.save("defect_dataset_2", dataset)
