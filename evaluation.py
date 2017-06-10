import cv2
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
from segmentation import *
import re

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)

    return l

def get_bounding_rect(mask):
    cnts = cv2.findContours(mask.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[1]
    contours = cnts[0]
    x,y,w,h = cv2.boundingRect(contours)
    return (x,y,w,h)



if __name__ == '__main__':

    sequences = ['juice','iceskater']
    dataset_dir = "vot2013"
    segmentation_results_dir = "vot2013_segmentation_clustered_results"
    image_dir = os.listdir(dataset_dir)
    reinitialize_threhold = 0.4
    for image_seq_dir in image_dir:
        if image_seq_dir in sequences:
                print('Now evaluating {} video sequence'.format(image_seq_dir))
                #images_list = os.listdir(dataset_dir+'/'+image_seq_dir)
                seg_list =  os.listdir(os.path.join(segmentation_results_dir,image_seq_dir))
                segmented_masks_list = sort_nicely([binary_mask for binary_mask in seg_list if binary_mask.startswith('binary_mask_for_frame_')])
                #print(segmented_masks_list)
                gt_bb_labels_file = open(os.path.join(dataset_dir,image_seq_dir,'groundtruth.txt'), 'r')
                gt_bb_labels = []
                for line in gt_bb_labels_file:
                    gt_bb_labels.append([int(float(num)) for num in line[:-1].split(',')])
                overlap_ratio_sum = 0
                num_frames = 0
                robustness = 0
                for i,image in enumerate(segmented_masks_list):
                    segmented_mask = cv2.imread(os.path.join(segmentation_results_dir,image_seq_dir,image),0)
                    if segmented_mask is not None:
                        num_frames += 1
                        x,y,w,h = get_bounding_rect(segmented_mask)
                        x_act,y_act,w_act,h_act = tuple(gt_bb_labels[i])
                        intersect_area = max(0, min(x+w,x_act+w_act) - max(x,x_act))*max(0, min(y+h,y_act+h_act) - max(y,y_act))
                        union_area = w*h + w_act*h_act - intersect_area
                        overlap_ratio = intersect_area/union_area
                        if overlap_ratio > reinitialize_threhold:
                            robustness += 1
                        overlap_ratio_sum += overlap_ratio
                avg_overlap = overlap_ratio_sum/num_frames
                robustness  = robustness/num_frames
                print('Tracker Results on {} video sequence: '.format(image_seq_dir))
                print('Average Overlap [Accuracy]: {}'.format(avg_overlap))
                print('Robustness normalized over number of frames {}'.format(robustness))
                plt.figure()
                plt.scatter(robustness,avg_overlap,marker='^',color='r')
                axes = plt.gca()
                axes.set_xlim([0,1])
                axes.set_ylim([0,1])
                plt.grid(color='k', linewidth=0.5, linestyle='--')
                plt.xlabel('Robustness (Normalized by number of frames)')
                plt.ylabel('Accuracy (Average Overlap)')
                plt.title('A-R plot for {} video sequence'.format(image_seq_dir))
                plt.savefig('a-r_plot_{}.png'.format(image_seq_dir))
                        #print('Percentage overlap between predicted and ground truth bounding boxes: {}'.format(overlap_ratio*100.0))
