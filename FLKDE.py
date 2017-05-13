import cv2
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt


def FLKDE(prev_frame,curr_frame,foreground_rect):

	height,width,_ = prev_frame.shape
	print(height,width)
	window_size = 15
	foreground_mask = np.zeros(prev_frame.shape[:2], np.uint8)
	output_mask = np.zeros(prev_frame.shape[:2], np.uint8)
	r = foreground_rect
	foreground_mask[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])] = 255
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

	fg_indices = np.where(output_mask == 255)
	min_x,min_y,max_x,max_y = (np.amin(fg_indices[1]),np.amin(fg_indices[0]),\
							   np.amax(fg_indices[1]),np.amax(fg_indices[0]))
	return (min_x,min_y,max_x-min_x+1,max_y-min_y+1)
	cv2.imshow('output_mask',output_mask)
	cv2.imshow('current_frame',curr_frame)
	cv2.waitKey(2000)			
	return (min_x,min_y,max_x-min_x+1,max_y-min_y+1)

	# cv2.destroyAllWindows()			


if __name__ == '__main__':

    cap = cv2.VideoCapture("car-overhead-1.avi")
    count  = 0
    prev_frame = None
    r = None
    while(cap.isOpened()):
       ret, frame = cap.read()
       if ret == True:
           count += 1
           current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
           if count > 1000:    
                 if count == 1001:
                   r = cv2.selectROI("First Frame", current_frame, False, False)
                 else:
                   r =  FLKDE(prev_frame,current_frame,r)          
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

