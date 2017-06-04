import sys
import numpy as np
import os
import scipy.misc
from scipy.optimize import least_squares
import math
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sfm_utils import *
import cv2
import glob
from detect import shiTomasiCornerDetector
from epipolar_utils import *
import random


def factorization_method(points_im1, points_im2):
    # TODO: Implement this method!
    D = np.vstack([points_im1.T[0:2], points_im2.T[0:2]])
    u,s,v = np.linalg.svd(D)
    u = u[:,[0,1,2]]
    s = s[0:3]
    v = v[0:3]
    M = np.matmul(u, (np.sqrt(np.diag(s))))
    S = np.matmul(np.sqrt(np.diag(s)),v)
    return S, M


def normalized_eight_point_alg(points1, points2):
    # TODO: Implement this method!
    mean1 = np.mean(points1, axis=0)
    mean2 = np.mean(points2, axis=0)
    s1 = np.sqrt(2/(np.mean(np.square(points1 - mean1)[:,[0,1]])*2))
    s2 = np.sqrt(2/(np.mean(np.square(points2 - mean2)[:,[0,1]])*2))
    T1 = np.array([s1,0,-s1*mean1[0],0,s1,-s1*mean1[1], 0, 0, 1])
    T1 = T1.reshape((3,3))
    T2 = np.array([s2,0,-s2*mean2[0],0,s2,-s2*mean2[1], 0, 0, 1])
    T2 = T2.reshape((3,3))
    normalized_points1 = np.array([np.matmul(T1,p) for p in points1])
    normalized_points2 = np.array([np.matmul(T2,p) for p in points2])
    F = lls_eight_point_alg(normalized_points1, normalized_points2)
    f = np.matmul(np.matmul(T2.T, F), T1)
    f = f/f[2][2]
    return f


def lls_eight_point_alg(points1, points2):
    # TODO: Implement this method!
    W = []
    for i in range(len(points1)):
        W.append(points1[i][0]*points2[i][0])
        W.append(points1[i][1]*points2[i][0])
        W.append(points2[i][0])
        W.append(points1[i][0]*points2[i][1])
        W.append(points1[i][1]*points2[i][1])
        W.append(points2[i][1])
        W.append(points1[i][0])
        W.append(points1[i][1])
        W.append(1)
    W = np.array(W).reshape((len(points1),9))
    U, S, V = np.linalg.svd(W)
    f = V[-1]
    f = f.reshape((3,3))
    u, s, v = np.linalg.svd(f)
    s[len(s)-1] = 0
    f = np.matmul(np.matmul(u,np.diag(s)),v)
    
    return f

'''
LINEAR_ESTIMATE_3D_POINT given a corresponding points in different images,
compute the 3D point is the best linear estimate
Arguments:
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    point_3d - the 3D point
'''
def linear_estimate_3d_point(image_points, camera_matrices):
    # TODO: Implement this method!
    A = []
    for i in range(len(image_points)):
        p = image_points[i]
        c_matrix = camera_matrices[i]
        A.append(p[0]*c_matrix[2] - c_matrix[0])
        A.append(p[1]*c_matrix[2] - c_matrix[1])
    A = np.array(A)
    _, _, v = np.linalg.svd(A)
    P = v[-1]
    P = P/P[-1]
    return P[0:3]

'''
REPROJECTION_ERROR given a 3D point and its corresponding points in the image
planes, compute the reprojection error vector and associated Jacobian
Arguments:
    point_3d - the 3D point corresponding to points in the image
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    error - the 2Mx1 reprojection error vector
'''
def reprojection_error(point_3d, image_points, camera_matrices):
    # TODO: Implement this method!
    e = []
    point_3d = np.hstack((point_3d,1))
    for i in range(len(image_points)):
        p = image_points[i]
        c_matrix = camera_matrices[i]
        p_estimate = c_matrix.dot(point_3d)
        p_estimate = p_estimate/p_estimate[-1]
        p_estimate = p_estimate[0:2]
        temp = p_estimate - p
        e.append(temp[0])
        e.append(temp[1])
    return e


'''
JACOBIAN given a 3D point and its corresponding points in the image
planes, compute the reprojection error vector and associated Jacobian
Arguments:
    point_3d - the 3D point corresponding to points in the image
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    jacobian - the 2Mx3 Jacobian matrix
'''
def jacobian(point_3d, camera_matrices):
    # TODO: Implement this method!
    J = []
    for c_matrix in camera_matrices:
        p = c_matrix.dot(np.hstack((point_3d,1)))
        denominator = np.square(p[2])
        J.append((p[2]*np.array([c_matrix[0][0:3]]) - p[0]*np.array([c_matrix[2][0:3]]))/denominator)
        J.append((p[2]*np.array([c_matrix[1][0:3]]) - p[1]*np.array([c_matrix[2][0:3]]))/denominator)
    J = np.array(J).reshape((2*len(camera_matrices),3))
    return J

'''
NONLINEAR_ESTIMATE_3D_POINT given a corresponding points in different images,
compute the 3D point that iteratively updates the points
Arguments:
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    point_3d - the 3D point
'''
def nonlinear_estimate_3d_point(image_points, camera_matrices, P = []):
    # TODO: Implement this method!
    if len(P) ==0:
        P = linear_estimate_3d_point(image_points, camera_matrices)
    J = jacobian(P, camera_matrices)
    e = reprojection_error(P, image_points, camera_matrices)
    #while any(error > 1 for error in e):
    for i in range(100):
    	J = jacobian(P, camera_matrices)
    	e = reprojection_error(P, image_points, camera_matrices)
    	P = P - np.linalg.inv(J.T.dot(J)).dot(J.T).dot(e)
    return P, e


def ransac_estimate_fundamental(points1, points2):
    max_inliners = 0
    num_points = len(points1)
    threshold = 0.01
    f = np.zeros((3,3))

    for i in range(100):
        num_inliners = 0
        errors = []
        index = random.sample(range(num_points), 8)
        sample1 = points1[index]
        sample2 = points2[index]
        f_temp = normalized_eight_point_alg(sample1, sample2)
        for i in range(num_points):
            error = np.abs(points1[i].dot(f_temp).dot(points2[i]))
            errors.append(error)
            if error <= threshold: num_inliners = num_inliners + 1
        if num_inliners > max_inliners:
            max_inliners = num_inliners
            f = f_temp
            print max_inliners
            #print errors
    print "got "+str(max_inliners) + " out of " + str(num_points) + " points"
    return f


if __name__ == '__main__':
    run_pipeline = False
    mask_image_dir = 'data/masks/cup/'
    original_image_dir = 'data/vot/cup/'
    lk_params = dict( winSize  = (25,25),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.05))

    # prepare input images (original * mask)
    mask_image0 = scipy.misc.imread(mask_image_dir+'binary_mask_for_frame_0.png')
    original_image0 = scipy.misc.imread(original_image_dir + '00000001.jpg', flatten = True)
    mask_image1 = scipy.misc.imread(mask_image_dir+'binary_mask_for_frame_40.png')
    original_image1 = scipy.misc.imread(original_image_dir + '00000041.jpg', flatten = True)
    im0 = mask_image0 * original_image0
    scipy.misc.imsave('data/multiply/m0.png', im0)
    im1 = mask_image1 * original_image1
    scipy.misc.imsave('data/multiply/m40.png', im1)
    image0 = 'data/multiply/m0.png'
    image1 = 'data/multiply/m40.png'
    im0 = cv2.imread(image0)
    im_gray0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
    im1 = cv2.imread(image1)
    im_gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)

    # find correspondances on two images
    pt0 = shiTomasiCornerDetector(im_gray0, (2,2), 600, 0.001, 5)
    pt0 = pt0.reshape(len(pt0),2)
    pt1, st, err = cv2.calcOpticalFlowPyrLK(im_gray0, im_gray1, pt0, None, **lk_params)
    assert len(pt0) == len(pt1)
    print pt0.shape

    # see if the detected points are good (uncomment to run it)
    '''
    for i in pt1:
        x,y = i
        cv2.circle(im1, (x,y), 3, (0,255,0), -1)
    # Show the cropped image
    cv2.imshow("im1", im1)
    cv2.waitKey(0)
    '''



    ## affine SFM (uncomment to run it)
    '''
    structure, motion  = factorization_method(pt0, pt1)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    scatter_3D_axis_equal(structure[0,:], structure[1,:], structure[2,:], ax)
    ax.set_title('Factorization Method')
    plt.show()
    '''


    ## bundle adjustment with projective SFM
    pt_0 = np.hstack((pt0,np.ones((len(pt0),1))))
    pt_1 = np.hstack((pt1,np.ones((len(pt1),1))))
    #F = normalized_eight_point_alg(pt_0, pt_1)
    F = ransac_estimate_fundamental(pt_0, pt_1)

    # calculate b for camera matrix M
    _,_,v = np.linalg.svd(F)
    b = v[-1:][0]
    b_cross = np.array([0,-b[2],b[1], b[2], 0, -b[0], -b[1], b[0], 0]).reshape((3,3))
    M = np.hstack((-b_cross.dot(F), b.reshape((3,1))))
    camera_matrices = np.zeros((2, 3, 4))
    camera_matrices[0, :, :] = np.hstack((np.eye(3), np.zeros((3,1))))  # canonical camera
    camera_matrices[1, :, :] = M
    
    # calculate 3D
    points_3d = []
    print structure.shape
    for i in range(len(pt0)):
        pts = np.vstack((pt0[i], pt1[i]))
        point_3d, e  = nonlinear_estimate_3d_point(pts, camera_matrices)
        if all(error < 0.5 for error in e):
            points_3d.append(point_3d)
    points_3d = np.array(points_3d)
    print "with "+str(len(points_3d)) + " total 3d points"
    print points_3d
    '''
    print points_3d.max()
    print points_3d.min()
    print points_3d.mean()
    print points_3d.std()
    '''

    # plot 3D
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(points_3d[:,0], points_3d[:,1], points_3d[:,2],
        c='k', depthshade=True, s=3)
    plt.show()