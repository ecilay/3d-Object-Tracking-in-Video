import cv2
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt


def small_region_removal(mask,threshold):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    result = np.zeros((output.shape),np.uint8)
    for i in range(0, nb_components):
        if sizes[i] >= threshold:
            result[output == i + 1] = 255

    return result

def FLKDE(prev_frame,curr_frame,foreground_mask):

	height,width,_ = prev_frame.shape
	window_size = 20
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

	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
	dilated_mask = small_region_removal(cv2.dilate(output_mask,kernel,iterations = 1),250)
	cv2.imshow('output_mask',dilated_mask)
	cv2.imshow('current_frame',curr_frame)
	cv2.waitKey(2000)
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
           current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
           if count > 1000:
                 if count == 1001:
                   r = cv2.selectROI("First Frame", current_frame, False, False)
                   foreground_mask = np.zeros(current_frame.shape[:2], np.uint8)
                   foreground_mask[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])] = 255
                 else:
                   foreground_mask =  FLKDE(prev_frame,current_frame,foreground_mask)
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
