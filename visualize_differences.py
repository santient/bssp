# Credit goes to Jonathan Roth for difference code

# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt

import sys

sys.path.append('/home/jr0th/github/segmentation/code/')

import skimage.io
import skimage.morphology
import skimage.segmentation

import sklearn.metrics

import os.path
import os

import numpy as np

import time

import glob

import pandas

import helper.metrics

import tqdm

import warnings
warnings.filterwarnings('ignore')

debug = True


# In[2]:

from pylab import rcParams
rcParams['figure.figsize'] = 5, 10


# In[3]:

mo_data_dir = "/data1/santiago/BBBC021/cnn-segmentation/labels/"
#mat_dir = '/home/jr0th/github/segmentation/experiments/DL_on_Hand_boundary_augment/' + tag + '/IoU/'
dif_img_out_dir = "/data1/santiago/BBBC021/differences/"


# In[4]:

cp_data_dir = "/data1/cp-segmentations/experiments/original/Outlines/"
#path_files_test = '/home/jr0th/github/segmentation/data/BBBC022/test.txt'


# In[5]:

#with open(path_files_test) as f:
#    test_files = f.read().splitlines()
cp_files = glob.glob(cp_data_dir+"*/*/*Nuclei.tiff")
cnn_files = glob.glob(mo_data_dir+"*/*.tif")
assert len(cp_files) == len(cnn_files)


# In[6]:

metadata = pandas.read_csv("/data1/santiago/BBBC021/metadata/metadata_rcnn.csv")
metadata = metadata.drop_duplicates(['Image_Metadata_Compound','Image_Metadata_Concentration'])
df = pandas.DataFrame(columns=['CP_Path','CNN_Path','Treatment','Concentration','Dif_Path','CP_Nuclei','CNN_Nuclei','Matches','Overdetections','Underdetections','Mean_IoU'])
nextrow = {}


# In[7]:

def dif_path(cnn_path):
    parent = cnn_path.split("/")[-2]
    basename = os.path.basename(cnn_path)
    path = dif_img_out_dir+parent
    if not os.path.exists(path):
        os.mkdir(path)
    return os.path.join(path,basename)


# In[8]:

def cp_path(cnn_path):
    parent = cnn_path.split("/")[-2]
    basename = os.path.splitext(os.path.basename(cnn_path))[0]+"_Nuclei.tiff"
    path = cp_data_dir+parent+"/"+parent
    return os.path.join(path,basename)


# In[ ]:

def visualize(mat, seg_cp, seg_model):

    # get number of nuclei
    nb_nuc_gt = mat.shape[0]
    nb_nuc_model = mat.shape[1]
    
    if debug:
        nextrow["CP_Nuclei"] = nb_nuc_gt
        nextrow["CNN_Nuclei"] = nb_nuc_model
        #print('# nuclei cell profiler', nb_nuc_gt)
        #print('# nuclei cnn model', nb_nuc_model)
    
    # only allow assignments if IoU is at least 0.5
    detection_map = (mat > 0.5)
    nb_matches = np.sum(detection_map)
    detection_map_gt = np.sum(detection_map, 1)
    detection_map_model = np.sum(detection_map, 0)
    
    # mask with matches
    detection_rate = mat * detection_map
    
    nb_overdetection = nb_nuc_model - nb_matches
    nb_underdetection = nb_nuc_gt - nb_matches
    
    if debug:
        nextrow["Matches"] = nb_matches
        nextrow["Overdetections"] = nb_overdetection
        nextrow["Underdetections"] = nb_underdetection
        #print('# matches', nb_matches)
        #print('# overdetections', nb_overdetection)
        #print('# underdetections', nb_underdetection)
        
    mean_IoU = np.mean(np.sum(detection_rate, axis = 1))
    
    if debug:
        nextrow["Mean_IoU"] = mean_IoU
        #print('# mean IoU', mean_IoU)
    
    # plot masked matrix
    #plt.figure(figsize=(10,10))
    #plt.matshow(detection_rate)
    #plt.show()
    
    # get indices of mislabeled cells
    error_underdetected = np.nonzero(detection_map_gt == 0)[0] + 1
    error_overdetected = np.nonzero(detection_map_model == 0)[0] + 1

    # get empty buffer image
    error_img = np.zeros((seg_model.shape[0], seg_model.shape[1], 3), dtype = np.ubyte)

    # color image
    brownish = [172, 128, 0]
    blueish = [31, 190, 214]
    for error in error_underdetected:
        # brownish: underdetected
        error_img[seg_cp == error, :] = brownish
    for error in error_overdetected:
        # blueish: overdetected
        error_img[seg_model == error, :] = error_img[seg_model == error, :] + blueish
        
    # pixels where under- and overdetection occured appear in pink
    
    return error_img

    


# In[ ]:

#files = filter(lambda f: len(glob.glob(mo_data_dir+"*/"+f)) == 0, metadata["Image_FileName_DAPI"])
for _, row in tqdm.tqdm(metadata.iterrows()):
    filename = mo_data_dir+row["Image_PathName_DAPI"]+row["Image_FileName_DAPI"] # cnn
    #filename_wo_ext = os.path.splitext(filename)[0]
    
    # load error matrices
    #IoU = np.load(mat_dir + filename_wo_ext + '.npy')
    
    # load segmentations
    seg_cp = skimage.io.imread(cp_path(filename))
    seg_mo = skimage.io.imread(filename)
    #seg_mo[seg_mo == 1] = 0
    [nb_overdetection, nb_underdetection, mean_IoU, IoU] = helper.metrics.compare_two_labels(seg_mo, seg_cp, True)
    # visualize only if matrices are full (no empty images and some detection)
    if(IoU.size != 0):
        nextrow["CP_Path"] = cp_path(filename)
        nextrow["CNN_Path"] = filename
        nextrow["Treatment"] = row["Image_Metadata_Compound"]
        nextrow["Concentration"] = row["Image_Metadata_Concentration"]
        nextrow["Dif_Path"] = dif_path(filename)
        error_img = visualize(IoU, seg_cp, seg_mo)
        assert len(nextrow) == 11
        df = df.append(nextrow, ignore_index=True)
        nextrow = {}
        #plt.imshow(error_img)
        #plt.show()
        skimage.io.imsave(dif_path(filename), error_img)
        df.to_csv(dif_img_out_dir + "differences.csv")


# In[ ]:



