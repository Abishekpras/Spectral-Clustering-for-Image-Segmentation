# coding : utf-8
#author : GE

import random
import copy
import numpy as np
import math

class LandMark:
    def __init__(self):
        self.matrix = np.eye(2)

    def getmatrix(self, matrix, k, density):
        self.matrix = matrix
        length_m = len(matrix)
        indexes = random.sample(range(length_m), (int)(length_m * density))
        U = []
        for i in indexes:
            U.append(copy.copy(self.matrix[i]))

        length_u = len(U)
        Z = np.zeros((length_m, length_u))
        for i in range(length_m):
            for j in range(length_u):
                Z[i,j] = self.gaussian(self.matrix[i],U[j])

        temp1 = Z.sum(axis=1)
        for i in range(length_m):
            Z[i] = Z[i] / temp1[i]

        temp2 = Z.sum(axis=0)
        D = np.zeros((length_u, length_u))
        for i in range(length_u):
            D[i,i] = 1 / math.sqrt(temp2[i])
        #print(Z)
        #temp = Z.sum(axis=1)
        #print(temp)
        #print(D)

        Z = np.dot(Z, D)

        #print(Z)
        W = np.dot(Z, Z.T)
        #print(W)
        #temp = W.sum(axis=1)
        #print(temp)

    def gaussian(self,x, u):
        norm = np.sum(np.square(x - u))
        return np.exp(-norm/2*1)

if __name__ == "__main__":
    lm = LandMark()
    map = np.mat([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]])
    lm.getmatrix(map, 1, 0.7)
