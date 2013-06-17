from numpy import *

def histogram2dEstimator(data, estx, esty, w):
    estimates = [[0.0 for j in range(len(estx[0]))] for i in range(len(estx))]
    for i in range(len(estx)):
        for j in range(len(estx[0])):
            for k in range(len(data[0])):
                if abs(estx[i][j] - data[0][k]) < (w/2) and abs(esty[i][j] - data[1][k]) < (w/2):
                    estimates[i][j] += 1
            estimates[i][j] = estimates[i][j] / (w ** 2) / len(data[0])
    return estimates

def kernel2dEstimator(data, estx, esty, w):
    estimates = [[0.0 for j in range(len(estx[0]))] for i in range(len(estx))]
    for i in range(len(estx)):
        for j in range(len(estx[0])):
            for k in range(len(data[0])):
                estimates[i][j] += exp(-((estx[i][j]-data[0][k])**2+(esty[i][j]-data[1][k])**2)/(2*(w**2)))
            estimates[i][j] = estimates[i][j] / ((2*pi*(w ** 2))**1) / len(data[0])
    return estimates

            
