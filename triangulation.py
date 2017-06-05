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


def factorization_method(D):
    # TODO: Implement this method!
    #D = np.vstack([points_im1.T[0:2], points_im2.T[0:2]])
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
            #print errors
    print "got "+str(max_inliners) + " out of " + str(num_points) + " points"
    return f


if __name__ == '__main__':
    category = 'juice'
    print "this will construct a " + category
    lk_params = dict( winSize  = (25,25),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.05))

    # prepare input images (original * mask) with num_images to use, in frame_counts
    frame_counts = [0,20,40] # choose the frames 


    im_grays = []
    num_images = len(frame_counts)
    for i in range(num_images):
        mask_image = scipy.misc.imread('data/mask/'+category+'/binary_mask_for_frame_'+str(frame_counts[i])+'.png')
        mask_image = mask_image/mask_image.max()
        image_num = '0'*(8-len(str(frame_counts[i]))) + str(frame_counts[i]+1)
        original_image = scipy.misc.imread('data/vot/'+category+'/'+image_num+'.jpg', flatten = True)
        im = mask_image * original_image
        scipy.misc.imsave('data/multiply/'+category+'/m'+str(frame_counts[i])+'.png', im)
        im = cv2.imread('data/multiply/'+category+'/m'+str(frame_counts[i])+'.png')
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im_grays.append(im)
    im_grays = np.array(im_grays)

    # find correspondances on above images
    pt0 = shiTomasiCornerDetector(im_grays[0],(6,6), 100, 0.3, 7)
    pt0 = pt0.reshape(len(pt0),2)
    pts = []
    pts.append(pt0)
    for i in range(1,num_images):
        pt, st, err = cv2.calcOpticalFlowPyrLK(im_grays[i-1], im_grays[i], pts[i-1], None, **lk_params)
        pts.append(pt)
    pts = np.array(pts)
    num_detections = pts[-1].shape[0]
    print "detected "+str(num_detections)+" points"

    # see if the detected points are good (uncomment to run it)
    '''
    view_image = num_images -1
    im = im_grays[view_image]
    for p in pts[view_image]:
        x,y = p
        cv2.circle(im, (x,y), 3, (255, 0,0), -1)
    cv2.imshow("im", im)
    cv2.waitKey(0)
    '''

    ## affine SFM (uncomment to run it)
    '''
    D = np.zeros((0,num_detections))
    for pt in pts:
        D = np.vstack([D, pt.T])
    print D.shape
    structure, motion  = factorization_method(D)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    scatter_3D_axis_equal(structure[0,:], structure[1,:], structure[2,:], ax)
    ax.set_title('Factorization Method for a '+ category)
    plt.show()
    '''
    


    ## bundle adjustment with projective SFM
    # calculate f and M between pairs of cameras
    
    F =[]
    camera_matrices = np.zeros((num_images, 3, 4))
    camera_matrices[0, :, :] = np.hstack((np.eye(3), np.zeros((3,1))))  # canonical camera
    for i in range(1,num_images):
        pt_0 = pts[i-1]
        pt_1 = pts[i]
        pt_0 = np.hstack((pt_0,np.ones((len(pt_0),1))))
        pt_1 = np.hstack((pt_1,np.ones((len(pt_1),1))))
        #f = normalized_eight_point_alg(pt_0, pt_1)
        f = ransac_estimate_fundamental(pt_0, pt_1)
        if len(F) == 0:F.append(f)
        else: 
            f = F[-1].dot(f)
            f = f/f[2][2]
            F.append(f)
        # calculate b for camera matrix M between pair cameras

        _,_,v = np.linalg.svd(f)
        b = v[-1:][0]
        b_cross = np.array([0,-b[2],b[1], b[2], 0, -b[0], -b[1], b[0], 0]).reshape((3,3))
        M = np.hstack((-b_cross.dot(f), b.reshape((3,1))))
        camera_matrices[i, :, :] = M
    
    # calculate 3D
    points_3d = []
    for i in range(num_detections):
        pt_of_cameras = np.zeros((0,2))
        for pt in pts:
            pt_of_cameras = np.vstack((pt_of_cameras, pt[i]))
        point_3d, e  = nonlinear_estimate_3d_point(pt_of_cameras, camera_matrices)
        #print "error:"
        #print e
        #if all(np.abs(error) < 15 for error in e):
        if all(isinstance(x, (int, long, float))for x in point_3d):
            points_3d.append(point_3d)
    points_3d = np.array(points_3d)
    # remove outlier points
    print "got "+str(len(points_3d)) + " total 3d points: "
    print points_3d
    median_points_3d = np.array([np.median(points_3d[:,0]),np.median(points_3d[:,1]),np.median(points_3d[:,2])])
    points_3d_valid = []
    diff_norms = []
    for p in points_3d:
        diff_norm = np.linalg.norm(p - median_points_3d)
        diff_norms.append(diff_norm)
    diff_norm_mean = np.median(diff_norms)
    diff_norm_std = np.std(diff_norms)
    print "median: "+str(diff_norm_mean)
    print "std: "+str(diff_norm_std)
    for i in range(len(points_3d)):
        diff_norm = diff_norms[i]
        if diff_norm <= diff_norm_mean+0*diff_norm_std:
            points_3d_valid.append(points_3d[i])
    points_3d_valid = np.array(points_3d_valid)
    print "got "+str(len(points_3d_valid))+" valid 3d points:"

    
    #print points_3d.max()
    #print points_3d.min()
    #print points_3d.mean()
    #print points_3d.std()
    

    # plot 3D
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(points_3d_valid[:,0], points_3d_valid[:,1], points_3d_valid[:,2],
        c='k', depthshade=True, s=3)
    plt.show()
    
