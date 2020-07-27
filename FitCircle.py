#!/usr/bin/env python
# coding: utf-8


import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


data = np.load('FK_data.npz')
data1 = data['FK_pose'][0]
target1 = [data1[i,0:3,3] for i in range(20)]
A1 = np.array(target1).T


def fitplane(A1, isPlot=False):
# fit the plane
    n = A1.shape[1]
    x_cen = np.mean(A1, axis=1).reshape(3, 1)
    A1_cen = A1 - x_cen
    k = np.max(abs(A1_cen))
    A1_prime = A1_cen/k
    U1, _, _ = np.linalg.svd(A1_prime)
    W1 = np.eye(3)
    W1[2,2] = np.linalg.det(U1)
    B1 = W1@U1
    n1 = B1[:, 2].reshape(3,1)
    P1 = B1[:, 0:2].swapaxes(0, 1)
    A2D1 = P1@A1_cen

# estimate circle
    A2D1_prime = np.r_[2*A2D1, np.ones((1,n))]
    b1 = (A2D1[0, :]**2 + A2D1[1, :]**2).reshape(n, 1)
    c1 = np.linalg.inv(A2D1_prime @ A2D1_prime.T)@ A2D1_prime @ b1
    r = math.sqrt(c1[0]**2 + c1[1]**2 + c1[2])
    
# return to 3D plane
    c1[2] = 0
    center1 = B1@c1 + x_cen
    
# correct the sign of circular features
    sign1 = np.cross((A1[:,0] - center1.reshape(1,3)),(A1[:,1] - center1.reshape(1,3)))@ n1
    if sign1 < 0:
        n1 = -n1    
    
    err = 0
    for i in range(n):
        err = err + (np.linalg.norm(A1[:,i].reshape(3,1)-center1) - r)**2
    RMSE = math.sqrt(err/n)
    print('CIRCLE_RMSE: {}'.format(RMSE))

# plot
    if isPlot:
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # sphere
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = np.cos(u)*np.sin(v)*r + center1[0]
        y = np.sin(u)*np.sin(v)*r + center1[1]
        z = np.cos(v)*r + center1[2]
        ax.plot_wireframe(x, y, z, color="r")

        for i in range(20):
            ax.scatter(A1[:,i][0],A1[:,i][1],A1[:,i][2])
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()
    return n1, center1, r

n1, center1, r = fitplane(A1, isPlot=True)