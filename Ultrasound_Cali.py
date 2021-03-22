import numpy as np
import scipy
from scipy.optimize import curve_fit
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

uv_array = []
localhost = '192.168.3.3'

class UltrasoundCalibration():
    def __init__(self):
        self.T_C = np.eye(4)  # fixed
        self.P_T = np.eye(4)  # fixed
        self.FB = np.zeros((2, 4))
        self.UV = np.zeros((4, 10))
        self.phai = np.zeros((10, 1))

    def UV_Generator(self, u, v):
        return np.array([
            [u, 0, 0, v, 0, 0, 1, 0, 0, 0],
            [0, u, 0, 0, v, 0, 0, 1, 0, 0],
            [0, 0, u, 0, 0, v, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        ])

    def getLine(self, point_a, point_b):
        x1, y1, z1 = point_a
        x2, y2, z2 = point_b
        a1, b1, c1, d1 = y2-y1, x1-x2, 0, x2*y1-x1*y2
        a2, b2, c2, d2 = 0, z2-z1, y1-y2, y2*z1-y1*z2
        return [
            np.array([a1, b1, c1, d1])/np.linalg.norm(np.array([a1, b1, c1])).tolist(),
            np.array([a2, b2, c2, d2])/np.linalg.norm(np.array([a2, b2, c2])).tolist()
        ]

    def getLines(self, line_num):
        if os.path.exists('params.npy'):
            os.remove('params.npy')
        
        params = []
        needle_tips = []
        for i in range(line_num):
            print('the {} th line'.format(i+1))
            input()
            needle_tip_1 = DataLoader.read_needle()
            
            input()
            needle_tip_2 = DataLoader.read_needle()
            param = self.getLine(needle_tip_1[0:3], needle_tip_2[0:3])
            params.append(param)
            
            needle_tips.append(np.vstack((needle_tip_1, needle_tip_2)))

        params = np.array(params)
        needle_tips = np.array(needle_tips)
        np.save('params.npy', params)
        np.save('needle_tips.npy', needle_tips)
        print("SAVED")

    def getUltrasound(self, times):
        if os.path.exists('data.npy'):
            os.remove('data.npy')
        if os.path.exists('pixel.npy'):
            os.remove('pixel.npy')
        for i in range(times):
            print('the {} th img'.format(i+1))
            input()
            if os.path.exists('data.npy'):
                data = np.load('data.npy')
                data = data.tolist()
                print('data length: ', len(data))
            else:
                data = []
            data.append(DataLoader.read_ultrasound())
            data = np.array(data)
            np.save('data.npy', data)

            # read pixel
            import cl_imgzmq
            cl_imgzmq.main(localhost)



colors = ['purple', 'red', 'green', 'blue', 'grey']

def workflow():
    data = np.load('data.npy')
    params = np.load('params.npy')
    
    data_num = len(data)  # image num

    if os.path.exists('pixel.npy'):
        uv_array = np.load('pixel.npy')
    else:
        uv_array = []
        
    X = []
    for i in range(data_num):  # image num
        for j in range(len(uv_array[0])):  # point num in each image
            x = (params[j+1] @ np.mat(data[i, 4:, :]).I @ data[i, 0:4, :] @ cali.UV_Generator(uv_array[i][j][0], uv_array[i][j][1]))
            ### 0 = line_params * point_onLine_3D / np.sqrt(line_params) 归一化后的直线参数
            X.append(x)
    X = np.array(X)  # (40, 2, 10)
    
    # solution
    A = np.zeros((10, 10))
    for i in range(data_num*len(uv_array[0])):
        Ai = X[i]
        A += Ai.T @ Ai
    eig = np.linalg.eig(A)
    print("特征值：", eig[0])
    fai = eig[1][:, np.argmin(eig[0])]
    fai = fai * (1.0/fai[-1])  # let the last num=1

    # show and save
    print('sol 1 fai: ', fai)
    np.save('solution.npy', fai)

    # validation
    sum = []
    for i in X:
        sum.append(i @ fai)
    avg = np.average(sum, axis=0)
    print('精度: ', avg)

    ##############################################################################################################################
    # draw
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    points = [[] for i in range(data_num)]
    points_mean = []
    for i in range(data_num):  # image num
        for j in range(len(uv_array[0])):  # point num in each image
            #               box                 detector                           uv_pixel                             trans_matrix
            point = (np.mat(data[i, 4:, :]).I @ data[i, 0:4, :] @ cali.UV_Generator(uv_array[i][j][0], uv_array[i][j][1]) @ fai)
            points[i].append(np.array(point).squeeze())
            points_mean.append(np.array(point).squeeze())
    points = np.array(points)
    points_mean = np.array(points_mean)
    mean = np.mean(points_mean, axis=0)
    
    ax.scatter(points[:, 0, 0], points[:, 0, 1], points[:, 0, 2], c='red')
    ax.scatter(points[:, 1, 0], points[:, 1, 1], points[:, 1, 2], c='green')
    ax.scatter(points[:, 2, 0], points[:, 2, 1], points[:, 2, 2], c='blue')
    ax.scatter(points[:, 3, 0], points[:, 3, 1], points[:, 3, 2], c='grey')
    # ax.scatter(points[:, 4, 0], points[:, 4, 1], points[:, 4, 2], c='yellow')

    lines = np.load('needle_tips.npy')
    for i in range(len(lines)-1):
        ax.plot(lines[i+1, :, 0], lines[i+1, :, 1], lines[i+1, :, 2], c=colors[i+1])

    scale = 100
    plt.xlim((mean[0]-scale, mean[0]+scale))
    plt.ylim((mean[1]-scale, mean[1]+scale))
    ax.scatter(mean[0]-scale, mean[1]-scale, mean[2]-scale, c='black')
    ax.scatter(mean[0]-scale, mean[1]-scale, mean[2]+scale, c='black')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    # plt.axis('scaled')
    # plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    #############################################################################################

def Solver(cali, lines_array, frames_array):
    """
    eg:
    lines_array.shape = (lines_num, 2 points, xyz)=(lines_num, 2, 3)
    ={
        'A': [[x,y,z],[x,y,z]],
        'B': [[x,y,z],[x,y,z]],
        ...
    }

    frames_array[0] = {
        'matrix': [],  detector relative to box
        'A': [u,v],
        'B': [u,v],
        ...
    }
    """
    data_num = len(frames_array)  # image num
        
    X = []
    for i in range(data_num):  # image num
        frame = frames_array[i]
        for pixel in frame.keys():  # point num in each image
            if pixel == 'matrix':
                continue
            params = cali.getLine(lines_array[pixel][0], lines_array[pixel][1])
            params = np.array(params)
            x = (params @ frame['matrix'] @ cali.UV_Generator(frame[pixel][0], frame[pixel][1]))
            X.append(x)
    X = np.array(X)  # (40, 2, 10)
    
    # solution 1
    A = np.zeros((10, 10))
    for i in range(len(X)):
        Ai = X[i]
        A += Ai.T @ Ai
    eig = np.linalg.eig(A)
    print("特征值：", eig[0])
    fai = eig[1][:, np.argmin(eig[0])]
    fai = fai * (1.0/fai[-1])  # let the last num=1

    # show and save
    print('sol 1 fai: ', fai)
    np.save('solution.npy', fai)

    # validation
    sum = []
    for i in X:
        sum.append(i @ fai)
    avg = np.average(sum, axis=0)
    print('精度: ', avg)

    return fai, avg


def faiReshaper(fai):
    newFai =  np.mat([
        [fai[0], fai[3], fai[6]],
        [fai[1], fai[4], fai[7]],
        [fai[2], fai[5], fai[8]],
        [0, 0, 1]
    ])

    return newFai


def show2D(point_3D):
    fai = np.load('solution.npy').flatten().tolist()
    data = np.load('data.npy')
    fai = faiReshaper(fai)
    point_3D = np.array(point_3D)
    
    # point_3D = np.array([-60.74370637, 24.20156662, 65.91986152, 1.0]).reshape(4, 1)  # (236, 160)
    # print(np.mat(data[0, 4:, :]).I @ data[0, 0:4, :] @ fai @ np.array([236, 160, 0 ,1]))
    point_uv = (fai.T @ fai).I @ fai.T @ np.mat(data[0, 0:4, :]).I @ np.mat(data[0, 4:, :]) @ point_3D
    point_uv = np.array(point_uv+0.5).astype('int').flatten()[0:-1]
    print(point_uv)


if __name__ == "__main__":
    cali = UltrasoundCalibration()
    # cali.getLines(5)  # calc 3D line equation
    cali.getUltrasound(10)  # get detector pose and box pose with pixels
    workflow()
    
    # show2D(point_3D=[-60.74370637, 24.20156662, 65.91986152, 1.0])