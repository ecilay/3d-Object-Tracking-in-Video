import cv2
import numpy as np

from scipy import interpolate
from scipy import ndimage
from scipy.signal import convolve2d

def reducePyramid(image, pyramid):
	out = cv2.GaussianBlur(image, (5,5), 1, 1)
	return out[0:pyramid.shape[0]:2, 0:pyramid.shape[1]:2]
	
def generatePyramid(image, level):
	pyramid = [image]
	for i in np.arange(1, level):
		pyramid.append(reducePyramid(image, pyramid[int(i) - 1]))
	return pyramid

def calcXDerivative(img, center):
	x = center[0]
	y = center[1]
	x_minus = max(x - 1, 0)
	x_plus = min(x + 1, img.shape[1] -1)
	return (img[y, x_plus] - img[y, x_minus]) / 2

def calcYDerivative(img, center):
	x = center[0]
	y = center[1]
	y_minus = max(y - 1, 0)
	y_plus = min(y + 1, img.shape[0] -1)
	return (img[y_plus, x] - img[y_minus, x]) / 2

def LKPyramidTracking(imgA, imgB, points):
	winR = 20 
	th = 0.01
	maxIterations = 100
	minImageSize = 32 
	maxLevels = int(np.floor(np.log2(np.min(imgA.shape) / minImageSize)))
	imgA = imgA.astype(np.float64)
	imgB = imgB.astype(np.float64)
	pyramidA = generatePyramid(imgA, maxLevels)
	pyramidB = generatePyramid(imgB, maxLevels)
	results = []
	for point in points:
		point = point[0]
		new_point = np.copy(point)
		point = np.array([int(point[0]), int(point[1])])
		guess = np.zeros((2))
		final_flow = None
		for level in range(maxLevels - 1, -1, -1):
			lPoint = point / (2 ** level)
			v = np.zeros((2))
			A = np.zeros((2,2))
			pyrA = pyramidA[level]
			xDeriv = convolve2d(pyrA, np.array([[-1, 1],[-1,1]]), 'valid')
			yDeriv = convolve2d(pyrA, np.array([[-1, -1],[1,1]]), 'valid')
			for y in range(max(0, lPoint[1] - winR), min(lPoint[1] + winR, pyrA.shape[0] - 1)):
				for x in range(max(0, lPoint[0] - winR), min(lPoint[0] + winR, pyrA.shape[1] - 1)):
					dIx = xDeriv[y,x]#calcXDerivative(pyrA, (x,y))
					dIy = yDeriv[y,x]#calcYDerivative(pyrA, (x,y))
					dIxy = dIx * dIy				
					A[0,0] += dIx ** 2
					A[1,1] += dIy ** 2
					A[1,0] += dIxy
					A[0,1] += dIxy

			for k in range(maxIterations):
				b = np.zeros(2)
				for y in range(max(0, lPoint[1] - winR), min(lPoint[1] + winR, pyrA.shape[0] - 1)):
					for x in range(max(0, lPoint[0] - winR), min(lPoint[0] + winR, pyrA.shape[1] - 1)):
						x_next = int(lPoint[0] + guess[0] + v[0])
						y_next = int(lPoint[1] + guess[1] + v[1])
						dIt = pyramidB[level][y_next, x_next] - pyrA[lPoint[1], lPoint[0]]
						dIx = xDeriv[y,x] #calcXDerivative(pyrA, (x,y))
						dIy = yDeriv[y,x] #calcYDerivative(pyrA, (x,y))
						b[0] += dIx * dIt
						b[1] += dIy * dIt
				flow = np.linalg.lstsq(A,b)[0] # should b be negative?
				v += flow
			final_flow = v
			guess = 2 * (guess + final_flow)
		final_flow = guess + final_flow
		new_point = point + final_flow
		results.append([new_point])	
	return results

fourcc = cv2.cv.FOURCC(*'DIVX')
v = cv2.VideoWriter("test.avi", fourcc, 24, (320,240))
num_frames = 140 
features = None
img = None
img_grey = None
img_next = None
img_next_grey = None
lk_params = dict( winSize  = (15,15),
                  maxLevel = 1,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
for i in range(num_frames):
	print "Frame " + str(i + 1)
	if i == 0:
		image_name = "juice/" + str(i+1) + ".jpg"
		img = cv2.imread(image_name)
		img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		features = cv2.goodFeaturesToTrack(img_grey, 1, 0.01, 5)
	show_num = True
	for f in features:
		p = (int(f[0][0]), int(f[0][1]))
		cv2.circle(img, p, 3, (255, 0, 0), -1)
	output = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	v.write(output)
	#cv2.imshow(str(i + 1), img)
	#cv2.waitKey(0)	
	if i + 1 == num_frames:
		break

	img_next = cv2.imread("juice/" + str(i+2) + ".jpg")
	img_next_grey = cv2.cvtColor(img_next, cv2.COLOR_BGR2GRAY)
	features = LKPyramidTracking(img_grey, img_next_grey, features)
	img = img_next
	img_grey = img_next_grey
cv2.destroyAllWindows()
v.release()
cv2.waitKey(1000)
