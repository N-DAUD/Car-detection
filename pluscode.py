import cv2
import numpy as np
from PIL import Image
from skimage.measure import label
from features import *


def convert_colorspace(img, color_space='RGB'):
    if color_space == 'RGB':
        feature_image = np.copy(img)      
    elif color_space == 'HSV':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif color_space == 'YCrCb':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        raise ValueError('Color space is not found.')
        
    return feature_image


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
  
    imcopy = np.copy(img)
 
    for bbox in bboxes:
     
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
   
    return imcopy


def add_heat(heatmap, bbox_list):
   
    for box in bbox_list:
      
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap


def apply_threshold(heatmap, threshold):
    
    heatmap[heatmap < threshold] = 0
    heatmap[heatmap > 0] = 1
    
    return heatmap


def draw_labeled_bboxes(img, labels):
    box_list = []
    

    for car_number in range(1, labels[1]+1):
    
        nonzero = (labels[0] == car_number).nonzero()
      
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
     
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        box_list.append(bbox)

    # Draw boxes on the image
    out_img = draw_boxes(img, box_list, color=(0, 0, 255), thick=6)
        
    return out_img


def extract_features(imgs, color_space='RGB', spatial_size=(32, 32), hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    

    features = []
    

    for image in imgs:
        file_features = []

        feature_image = convert_colorspace(image, color_space)
            
        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
            
        if hist_feat == True:
      
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
            
        if hog_feat == True:
     
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
                
          
            file_features.append(hog_features)
        
        if len(file_features) == 0:
            raise ValueError('Feature vector is empty.')
            
        features.append(np.concatenate(file_features))
        
   
    return features


def find_cars(img, svc, X_scaler, y_start_stops, scales, window, color_space, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat):
    
    box_list = []
    
    for scale, (y_start, y_stop) in zip(scales, y_start_stops):
        img_tosearch = img[y_start:y_stop,:,:]
       
        if scale != 1:
            imshape = img_tosearch.shape
            img_tosearch = cv2.resize(img_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

        nxsteps = (img_tosearch.shape[1] - window) // pix_per_cell + 1
        nysteps = (img_tosearch.shape[0] - window) // pix_per_cell + 1

        for xc in range(nxsteps):
            for yc in range(nysteps):
                xleft = xc * pix_per_cell
                ytop = yc * pix_per_cell

           
                subimg = img_tosearch[ytop:ytop+window, xleft:xleft+window]

              
                test_features = X_scaler.transform(extract_features([subimg], 
                                                                    color_space, spatial_size,
                                                                    hist_bins, orient,
                                                                    pix_per_cell, cell_per_block,
                                                                    hog_channel, spatial_feat,
                                                                    hist_feat, hog_feat))
                
                test_prediction = svc.predict(test_features)

                if test_prediction[0] == 1:
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    bbox = ((xbox_left, ytop_draw+y_start), (xbox_left+win_draw, ytop_draw+win_draw+y_start))
                    box_list.append(bbox)
                    
        out_img = draw_boxes(img, box_list, color=(0, 0, 255), thick=6)
                    
    return out_img, box_list