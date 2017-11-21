# coding : utf-8
#author : GE

import random
import copy
import numpy as np
import math

class LandMark:
    def __init__(self):
        self.matrix = np.eye(2)

    def getmatrix(self, matrix, k):
        self.matrix = matrix
        length_m = len(matrix)
        indexes = random.sample(range(length_m), (int)(length_m * 0.7))
        U = []
        for i in indexes:
            U.append(copy.copy(self.matrix[i]))
        length_u = len(U)
        Z = np.zeros((length_m, length_u))
        for i in range(length_m):
            for j in range(length_u):
                Z[i,j] = self.gaussian(self.matrix[i],U[j])
        temp = Z.sum(axis= 1)
        D = np.zeros((length_m, length_m))
        for i in range(length_m):
            Z[i] = Z[i] / temp[i]
            D[i,i] = 1 / math.sqrt(temp[i])
        Z = np.dot(D, Z)
        W = np.dot(Z.T, Z)

    def gaussian(self,x, u):
        norm = np.linalg.norm(x-u, ord = 2)
        return np.exp(-norm/2*1)

if __name__ == "__main__":
    lm = LandMark()
    map = np.mat([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
    lm.getmatrix(map,1)
