import cv2
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt


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

def FLKDE(prev_frame,curr_frame,foreground_mask,count):

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
                print(hist[0].dtype,neighbourhood_mask.dtype); actual_mask = cv2.bitwise_and(hist[0],neighbourhood_mask)
                for ch in range(3):
                    hist[1][ch] = cv2.calcHist([prev_frame],[ch],actual_mask,[32],[0,256])
                    R,G,B = curr_frame[i,j,:]%32
                    scores[num] =  hist[1][R,G,B]
            if scores[0]>scores[1]:
                output_mask[i,j] = 255

    temp = cv2.dilate(output_mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10)),iterations = 1)
    #cv2.imshow('after only dilation',temp)
    #temp_2 = cv2.erode(temp,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2)),iterations = 1)
    #cv2.imshow('after dilation+erosion',temp_2)
    dilated_mask = small_region_removal(temp)
    final_mask = cv2.addWeighted(cv2.cvtColor(current_frame,cv2.COLOR_BGR2GRAY),0.8,dilated_mask,0.2,0)
    cv2.imshow('frame #{}'.format(count),final_mask)
    #cv2.imshow('after dilation+erosion+small_region_removal',dilated_mask)
    return dilated_mask


if __name__ == '__main__':

    cap = cv2.VideoCapture("car-overhead-1.avi")
    count  = 0
    prev_frame = None
    foreground_mask = None
    r = None
    while(cap.isOpened()):
       ret, frame = cap.read()
       if ret == True:
           count += 1
           current_frame = frame#cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
           if count > 1050:
                 if count == 1051:
                   r = cv2.selectROI("First Frame", current_frame, False, False)
                   foreground_mask = np.zeros(current_frame.shape[:2], np.uint8)
                   foreground_mask[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])] = 255
                 else:
                   foreground_mask =  FLKDE(prev_frame,current_frame,foreground_mask,count)
           else:
           		continue
           if cv2.waitKey(10) & 0xFF == ord('q'):
              break
           prev_frame  = current_frame
       else:
              break


    # # When everything done, release the captur
    cap.release()
    cv2.destroyAllWindows()
