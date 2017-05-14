import cv2
import numpy as np
from scipy.stats import multivariate_normal


def shiTomasiCornerDetector(im, blockDim, maxPoints, quality=0.01, distThreshold=10):
    grad_x = cv2.Sobel(im, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(im, cv2.CV_64F, 0, 1, ksize=5)

    grad_x2 = np.square(grad_x)
    grad_y2 = np.square(grad_y)
    grad_xy = np.multiply(grad_x, grad_y)

    result_matrix = np.zeros(im.shape)

    print "Calulating corners..."
    for x, y in np.ndindex(im.shape):
        x_min = max(0, x - int(blockDim[0] / 2))
        x_max = min(im.shape[1], x + int(blockDim[0] / 2)) # Remember, coordinates are (y,x) for images
        y_min = max(0, y - int(blockDim[1] / 2))
        y_max = min(im.shape[0], y + int(blockDim[1] / 2))

        M = np.zeros((2,2))
        for u in np.arange(x_min, x_max):
            for v in np.arange(y_min, y_max):
                M_intermediate = np.array([[grad_x2[v,u], grad_xy[v,u]],
                                           [grad_xy[v,u], grad_y2[v,u]]])
                M += M_intermediate
        eig = np.linalg.eig(M)[0]
        result_matrix[x, y] = np.min(eig)

    print "Collecting corners!"
    corners = []
    for x, y in np.ndindex(im.shape):
        if result_matrix[x, y] >= quality:
            corners.append((x, y))
    
    # Sort the valid corners by 
    print "Sorting corners!"
    corners.sort(key=lambda cord:result_matrix[cord], reverse=True)
    selected = [[corners.pop(0)]]

    def distance(a, b):
        a = np.array(a)
        b = np.array(b)
        return np.sqrt(np.sum((a - b) ** 2))

    for corner in corners:
        distance_calc = lambda c: distance(corner, c)
        if min(map(distance_calc, selected)) >= distThreshold:
            selected.append([corner])
        if len(selected) == maxPoints:
            break
    selected = np.array([np.array([np.array([x[0][0], x[0][1]], dtype=np.float32)]) for x in selected])
    return selected

if __name__ == '__main__':
    # Load the image
    im = cv2.imread("image.jpg")
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
  
    corners = shiTomasiCornerDetector(im_gray, (3,3), 25, 0.04, 10)
    corners_cv2 = cv2.goodFeaturesToTrack(im_gray, 25, 0.04, 10)
    for i in corners:
        x,y = i
        cv2.circle(im, (x,y), 3, (0,255,0), -1)
    print len(corners)
    # Show the cropped image
    cv2.imshow("Mine", im)
    cv2.waitKey(0)

    for i in corners_cv2:
        x,y = i.ravel()
        cv2.circle(im, (x,y), 3, (255,0,0), -1)
    print corners_cv2.shape
    cv2.imshow("CV2", im)
    cv2.waitKey(0)
