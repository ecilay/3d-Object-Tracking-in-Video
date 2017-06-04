import cv2
import numpy as np
import scipy.ndimage as sp

def shiTomasiCornerDetector(im, blockDim, maxPoints, quality, distThreshold):
    grad_y = cv2.Sobel(im, cv2.CV_64F, 1, 0, ksize=5)
    grad_x = cv2.Sobel(im, cv2.CV_64F, 0, 1, ksize=5)

    grad_x2 = np.square(grad_x)
    grad_y2 = np.square(grad_y)
    grad_xy = np.multiply(grad_x, grad_y)

    result_matrix = np.zeros(im.shape)

    for y, x in np.ndindex(im.shape):
        x_min = max(0, x - int(blockDim[0] / 2))
        x_max = min(im.shape[1], x + int(blockDim[0] / 2)) # Remember, coordinates are (y,x) for images
        y_min = max(0, y - int(blockDim[1] / 2))
        y_max = min(im.shape[0], y + int(blockDim[1] / 2))

        M = np.zeros((2,2))
        for v in np.arange(y_min, y_max):
            for u in np.arange(x_min, x_max):
                M_intermediate = np.array([[grad_x2[v,u], grad_xy[v,u]],
                                           [grad_xy[v,u], grad_y2[v,u]]])
                M += sp.filters.gaussian_filter(M_intermediate, 1, 0)
        eig = np.linalg.eig(M)[0]
        result_matrix[y, x] = np.min(eig)

    corners = []
    for y, x in np.ndindex(im.shape):
        if result_matrix[y, x] >= quality:
            corners.append((x, y))
    
    # Sort the valid corners by 
    corners.sort(key=lambda cord:result_matrix[(cord[1], cord[0])], reverse=True)
    selected = [[corners.pop(0)]]

    def distance(a, b):
        a = np.array(a)
        b = np.array(b[0])
        return np.sqrt(np.sum((a - b)**2))

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
    im = cv2.imread("image.png")
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
  
    corners = shiTomasiCornerDetector(im_gray, (3,3), 25, 0.01, 10)
    corners_cv2 = cv2.goodFeaturesToTrack(im_gray, 25, 0.04, 10)
    for i in corners:
        x,y = i[0]
        cv2.circle(im, (x,y), 3, (0,255,0), -1)
    # Show the cropped image
    cv2.imshow("Mine", im)
    cv2.waitKey(0)
    
    im2 = cv2.imread("image.png")
    for i in corners_cv2:
        x,y = i.ravel()
        cv2.circle(im2, (x,y), 3, (255,0,0), -1)
    cv2.imshow("CV2", im2)
    cv2.waitKey(0)
    