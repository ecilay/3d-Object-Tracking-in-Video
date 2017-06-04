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


def create_video(images_dir):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frames_per_sec = 10
    out = cv2.VideoWriter('output_{}.mp4'.format(images_dir),fourcc, 10, (320,240))
    frames = sort_nicely((os.listdir(images_dir)))
    for i,frame in enumerate(frames):
        print(frame)
        current_frame = cv2.imread(images_dir+'/'+frame)
        print('Processing frame {}'.format(i))
        out.write(current_frame)
    cv2.destroyAllWindows()
    out.release()

def get_bounding_rect(mask):
    cnts = cv2.findContours(mask.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[1]
    contours = cnts[0]
    x,y,w,h = cv2.boundingRect(contours)
    return (x,y,w,h)


if __name__ == '__main__':

    #if reading in a video sequence as in vot2013 dataset
    #create_video('video_sequence')
    #assert(False)
    sequences = ['juice']
    dataset_dir = "vot2013"
    image_dir = os.listdir(dataset_dir)
    for image_seq_dir in image_dir:
        if image_seq_dir in sequences:
            print('Now segmenting {} video sequence'.format(image_seq_dir))
            if not os.path.exists(dataset_dir+'_segmentation_clustered_results/'+image_seq_dir):
                os.makedirs(dataset_dir+'_segmentation_clustered_results/'+image_seq_dir)
            images_list = os.listdir(dataset_dir+'/'+image_seq_dir)
            gt_bb_labels_file = open(os.path.join(dataset_dir,image_seq_dir,'groundtruth.txt'), 'r')
            gt_bb_labels = []
            for line in gt_bb_labels_file:
                gt_bb_labels.append([int(float(num)) for num in line[:-1].split(',')])
            prev_frame = None
            overlap_ratio = -1
            for i,image in enumerate(images_list):
                if image[-3:] == 'jpg':
                    current_frame = cv2.imread(dataset_dir+'/'+image_seq_dir+'/'+image)
                    #current_frame = cv2.resize(current_frame, (0,0), fx=resize_factor, fy=resize_factor)
                    foreground_mask_current = cv2.imread(dataset_dir+'_segmentation_clustered_results/'+image_seq_dir+'/'+\
                                    'binary_mask_for_frame_{}.png'.format(i),0)
                    if i == 0 or overlap_ratio < 0.4:
                        if overlap_ratio < 0.4:
                            print('Reinitialization at frame {} [Overlap with ground truth < 40%]'.format(i))
                        try:
                            foreground_mask = grabcut(current_frame,tuple(gt_bb_labels[i]))
                            x,y,w,h = get_bounding_rect(foreground_mask)
                        except:
                            print('Reinitialization at frame {} [Grabcut error]'.format(i))
                            x,y,w,h = gt_bb_labels[i]
                            foreground_mask_current = np.zeros(current_frame.shape[:2],np.uint8)
                            foreground_mask_current[y:y+h,x:x+w] = 255
                            foreground_mask = foreground_mask_current
                            x,y,w,h = get_bounding_rect(foreground_mask)
                    elif foreground_mask_current == None:
                        try:
                            foreground_mask_coarse =  FLKDE(prev_frame,current_frame,foreground_mask,i)
                            cv2.imwrite(dataset_dir+'_segmentation_clustered_results/'+image_seq_dir+'/'+\
                                    'binary_mask_b4_grabcut_for_frame_{}.png'.format(i),foreground_mask_coarse)
                            x,y,w,h = get_bounding_rect(foreground_mask_coarse)
                            foreground_mask_current = grabcut(current_frame,(x,y,w,h))
                            foreground_mask = foreground_mask_current
                            x,y,w,h = get_bounding_rect(foreground_mask)
                        except:
                            try:
                                foreground_mask_current = grabcut(current_frame,tuple(gt_bb_labels[i]))
                                print('Reinitialization at frame {} [FLKDE error]'.format(i))
                                foreground_mask = foreground_mask_current
                                x,y,w,h = get_bounding_rect(foreground_mask)
                            except:
                                print('Reinitialization at frame {} [Grabcut error]'.format(i))
                                x,y,w,h = gt_bb_labels[i]
                                foreground_mask_current = np.zeros(current_frame.shape[:2],np.uint8)
                                foreground_mask_current[y:y+h,x:x+w] = 255
                                foreground_mask = foreground_mask_current
                                x,y,w,h = get_bounding_rect(foreground_mask)
                    # print('My output: ',x,y,w,h)
                    # print('Actual output: ',*(gt_bb_labels[i]))
                    x_act,y_act,w_act,h_act = tuple(gt_bb_labels[i])
                    intersect_area = max(0, min(x+w,x_act+w_act) - max(x,x_act)) * max(0, min(y+h,y_act+h_act) - max(y,y_act))
                    union_area = w*h + w_act*h_act - intersect_area
                    overlap_ratio = intersect_area/union_area
                    print('Percentage overlap between predicted and ground truth bounding boxes: {}'.format(overlap_ratio*100.0))
                    cv2.imwrite(dataset_dir+'_segmentation_clustered_results/'+image_seq_dir+'/'+\
                                'binary_mask_for_frame_{}.png'.format(i),foreground_mask)
                    final_mask = cv2.addWeighted(cv2.cvtColor(current_frame,\
                        cv2.COLOR_BGR2GRAY),0.3,foreground_mask,0.7,0)
                    cv2.imwrite(dataset_dir+'_segmentation_clustered_results/'+image_seq_dir+'/'+\
                                'blended_mask_for_frame_{}.png'.format(i),final_mask)
                    prev_frame  = copy.copy(current_frame)


    #if reading in an actual video (as in VISOR Dataset)

    # cap = cv2.VideoCapture("car-overhead-1.avi")
    # count  = 0
    # prev_frame = None
    # foreground_mask = None
    # r = None
    # while(cap.isOpened()):
    #    ret, frame = cap.read()
    #    if ret == True:
    #        count += 1
    #        current_frame = frame#cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #        if count > 1050:
    #              if count == 1051:
    #                r = cv2.selectROI("First Frame", current_frame, False, False)
    #                foreground_mask = np.zeros(current_frame.shape[:2], np.uint8)
    #                foreground_mask[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])] = 255
    #              else:
    #                foreground_mask =  FLKDE(prev_frame,current_frame,foreground_mask,count-1050)
    #        else:
    #        		continue
    #        if cv2.waitKey(10) & 0xFF == ord('q'):
    #           break
    #        prev_frame  = current_frame
    #    else:
    #           break


    # # When everything done, release the captur
    #cap.release()
    cv2.destroyAllWindows()
