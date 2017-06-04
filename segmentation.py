from __future__ import print_function
import cv2
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
import sys


BLUE = [255,0,0]        # rectangle color
RED = [0,0,255]         # PR BG
GREEN = [0,255,0]       # PR FG
BLACK = [0,0,0]         # sure BG
WHITE = [255,255,255]   # sure FG

DRAW_BG = {'color' : BLACK, 'val' : 0}
DRAW_FG = {'color' : WHITE, 'val' : 1}
DRAW_PR_FG = {'color' : GREEN, 'val' : 3}
DRAW_PR_BG = {'color' : RED, 'val' : 2}

# setting up flags
rect = (0,0,1,1)
drawing = False         # flag for drawing curves
rectangle = False       # flag for drawing rect
rect_over = False       # flag to check if rect drawn
rect_or_mask = 100      # flag for selecting rect or mask mode
value = DRAW_FG         # drawing initialized to FG
thickness = 3           # brush thickness


def small_region_removal(mask):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    result = np.zeros((output.shape),np.uint8)
    max_size = -1
    max_size_index = 1
    for i in range(0, nb_components):
        if sizes[i] >= max_size:
            max_size_index  = i
            max_size = sizes[i]
    result[output == max_size_index + 1] = 255

    return result


def grabcut(img,rect_):

    mask = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    cv2.grabCut(img,mask,rect_,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask2 = small_region_removal(np.where((mask==2)|(mask==0),0,1).astype('uint8')*255)
    return mask2

# def back_subtract_gmg():
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
#     fgbg = cv2.createBackgroundSubtractorGMG()

def FLKDE(prev_frame,curr_frame,foreground_mask,count):

    print('Now segmenting frame#{}'.format(count))
    height,width,_ = prev_frame.shape
    window_size = 15
    output_mask = np.zeros(prev_frame.shape[:2], np.uint8)
    foreground_hist = background_hist = np.zeros((32,32,32))

    for i in range(height):
        for j in range(width):
            neighbourhood_mask = np.zeros(prev_frame.shape[:2], np.uint8)
            neighbourhood_mask[max(0,i-window_size):min(i+window_size,height), \
            max(j-window_size,0):min(j+window_size,width)] = 255
            hist_type = [(foreground_mask,foreground_hist), \
            (cv2.bitwise_not(foreground_mask),background_hist)]
            scores = np.zeros(2)
            for num,hist in enumerate(hist_type):
                actual_mask = cv2.bitwise_and(hist[0],neighbourhood_mask)
                for ch in range(3):
                    hist[1][ch] = cv2.calcHist([prev_frame],[ch],actual_mask,[32],[0,256])
                    R,G,B = curr_frame[i,j,:]%32
                    scores[num] =  hist[1][R,G,B]
            if scores[0]>scores[1]:
                output_mask[i,j] = 255

    temp = cv2.dilate(output_mask,cv2.getStructuringElement(cv2.MORPH_RECT,(25,25)),iterations = 1)
    dilated_mask = small_region_removal(temp)

    return dilated_mask


## Not working as as of now. Needs to be degbugged
def FLKDE_efficient(prev_frame,curr_frame,foreground_mask,count,W = 15):

    print('Now segmenting frame#{}'.format(count))
    height,width,_ = prev_frame.shape
    output_mask = np.zeros(prev_frame.shape[:2], np.uint8)
    foreground_im = np.expand_dims(np.uint8(foreground_mask/255),axis = 2)*prev_frame
    background_im = np.expand_dims(np.uint8(cv2.bitwise_not(foreground_mask)/255),axis = 2)*prev_frame
    foreground_im  = np.lib.pad(foreground_im,((7,),(7,),(0,)),'constant')
    background_im  = np.lib.pad(background_im,((7,),(7,),(0,)),'reflect')
    foreground_hist,_ = np.histogramdd(foreground_im[:W,:W,:].reshape((-1,3)),\
            range = ((0,255),(0,255),(0,255)),bins = (32, 32, 32))
    background_hist,_ = np.histogramdd(background_im[:W,:W,:].reshape((-1,3)),\
            range = ((0,255),(0,255),(0,255)), bins = (32, 32, 32))
    RGB_val = curr_frame[0,0,:]//32
    if foreground_hist[RGB_val[0],RGB_val[1],RGB_val[2]] > background_hist[RGB_val[0],RGB_val[1],RGB_val[2]]:
        output_mask[0,0] = 255
    else:
        output_mask[0,0] = 0
    for row in range(0,height):
        if row%2 == 0:
            if row == 0:
                range_col = range(1,width)
            else:
                range_col = range(0,width)
        else:
            range_col = reversed(range(0,width))

        for j,col in enumerate(range_col):
                row_mod = row + W//2
                col_mod = col + W//2

                if (col == 0 and row%2 == 0) or (col == width - 1 and row%2 == 1):
                    #print('Adding: row {} and column {} to column {}'.format(row_mod+W//2,col_mod-W//2,col_mod+W//2+1))
                    #print('Subtracting: row {} and column {} to column {}'.format(row_mod-W//2-1,col_mod-W//2,col_mod+W//2+1))
                    forg_hist_add = foreground_im[row_mod+W//2,col_mod-W//2:col_mod+W//2+1,:]
                    forg_hist_subtract = foreground_im[row_mod-W//2-1,col_mod-W//2:col_mod+W//2+1,:]
                    back_hist_add = background_im[row_mod+W//2,col_mod-W//2:col_mod+W//2+1,:]
                    back_hist_subtract = background_im[row_mod-W//2-1,col_mod-W//2:col_mod+W//2+1,:]
                else:
                    if row%2 == 0:
                        #print('Adding: row {} to row {} and column {}'.format(row_mod-W//2,row_mod+W//2+1,col_mod+W//2))
                        #print('Subtracting: row {} to row {} and column {}'.format(row_mod-W//2,row_mod+W//2+1,col_mod-W//2-1))
                        forg_hist_add = foreground_im[row_mod-W//2:row_mod+W//2+1,col_mod+W//2,:]
                        forg_hist_subtract = foreground_im[row_mod-W//2:row_mod+W//2+1,col_mod-W//2-1,:]
                        back_hist_add = background_im[row_mod-W//2:row_mod+W//2+1,col_mod+W//2,:]
                        back_hist_subtract = background_im[row_mod-W//2:row_mod+W//2+1,col_mod-W//2-1,:]
                    else:
                        #print('Adding: row {} to row {} and column {}'.format(row_mod-W//2,row_mod+W//2+1,col_mod-W//2))
                        #print('Subtracting: row {} to row {} and column {}'.format(row_mod-W//2,row_mod+W//2+1,col_mod+W//2+1))
                        forg_hist_add = foreground_im[row_mod-W//2:row_mod+W//2+1,col_mod-W//2,:]
                        forg_hist_subtract = foreground_im[row_mod-W//2:row_mod+W//2+1,col_mod+W//2+1,:]
                        back_hist_add = background_im[row_mod-W//2:row_mod+W//2+1,col_mod-W//2,:]
                        back_hist_subtract = background_im[row_mod-W//2:row_mod+W//2+1,col_mod+W//2+1,:]

                ### Needs to be degbugged
                # Slow
                forg_add_indices = (forg_hist_add/(256/32)).astype(int)
                forg_subtract_indicies = (forg_hist_subtract/(256/32)).astype(int)
                back_add_indices = (back_hist_add/(256/32)).astype(int)
                back_subtract_indicies = (back_hist_subtract/(256/32)).astype(int)
                #
                foreground_hist[forg_add_indices] += 1
                foreground_hist[forg_subtract_indicies] -= 1
                background_hist[back_add_indices] += 1
                background_hist[back_subtract_indicies] -= 1

                RGB_val = (curr_frame[row,col,:]/(256/32)).astype(int)
                if foreground_hist[RGB_val[0],RGB_val[1],RGB_val[2]] > background_hist[RGB_val[0],RGB_val[1],RGB_val[2]]:
                    output_mask[row,col] = 255


    temp = cv2.dilate(output_mask,cv2.getStructuringElement(cv2.MORPH_RECT,(12,12)),iterations = 1)
    dilated_mask = small_region_removal(temp)
    cv2.imshow('after dilation+erosion+small_region_removal',dilated_mask)
    cv2.waitKey(0)
    return dilated_mask
