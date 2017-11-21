import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from numpy import linalg as la
from LandMark import LandMark


img = Image.open('/Users/Anirudh/Desktop/s.png')
img = img.convert('L')
plt.imshow(img)
img = np.asarray(img).astype(float)



a = {}
b = 0
for i in range(100):
    for j in range(100):
        a[b] = [i,j]
        b += 1
        

#similarity graph
A = np.zeros((10000,10000))


#taking sigma values = 2; and r=7
for i in range(10000):
    for j in range(10000):
        if i == j:
            A[i,j] = 1
        else:
            if A[i,j] != 0:
                A[j,i] = A[i,j]
            else:
                dist = pow((a[i][0]-a[j][0])**2+(a[i][1]-a[j][1])**2,0.5)
                if dist <= 7:
                    A[i,j] = np.exp((-(img[a[i][0],a[i][1]] - img[a[j][0],a[j][1]])**2-dist**2)/2.0)


lm = LandMark()
map = A
lm.getmatrix(map, 1, 0.01)



