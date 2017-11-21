import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from numpy import linalg as la
import time


#opening the image
img = Image.open('s.png')
#converting the image to grayscale
img = img.convert('L')

img = np.asarray(img).astype(float)/255.0
sh = img.shape
start = time.time()


#A dictionary that stores locations as the keys and node numbers as values
#e.g for image like [[1,2,3],[4,5,6],[7,8,9]] , pixel at location (0,0) -> 1 , (0,2) -> 3 , (1,0) -> 4..so on
a = {}
b = 0
for i in range(sh[0]):
    for j in range(sh[1]):
        a[(i,j)] = b
        b += 1
        
        
#similarity graph
A = np.zeros((sh[0]**2,sh[0]**2))
#distance threshold (k)
k = 7
#both sgima values choosen as 2
for l in range(sh[1]):
    for m in range(sh[0]):
        current = a[(l,m)]
        d=0
        flag = 0
        for j in range(m-k,m+k+1):
            for i in range(l-d,l+d+1): 
                #this makes sure that the indices are valid
                if (i,j) in a:
                    #if the value is already calculated
                    if A[a[(i,j)],current] != 0:
                        A[current,a[(i,j)]] = A[a[(i,j)],current]
                    else:
                        A[current,a[(i,j)]] = np.exp((-(img[l,m] - img[i,j])**2-((l-i)**2+(m-j)**2))/4.0)
            if j == m:
                flag = 1
            if flag == 1:
                d -= 1
            else:
                d += 1
end = time.time()
print("time taken:" + str(end-start)+"s")

#lm = LandMark()
#map = A
#lm.getmatrix(map, 1, 0.01)



