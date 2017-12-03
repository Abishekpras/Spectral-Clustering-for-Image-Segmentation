from numpy import linalg as la
import scipy.cluster.vq as vq
import scipy.ndimage as ndimage
import scipy as sp
import networkx as nx
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from sklearn.feature_extraction import image
from skimage.color import rgb2gray
import numpy as np
from scipy.spatial.distance import pdist, squareform

x, y = np.indices((100, 100))

center1 = (28, 24)
center2 = (40, 50)
center3 = (67, 58)
center4 = (24, 70)

radius1, radius2, radius3, radius4 = 16, 14, 15, 14

circle1 = (x - center1[0]) ** 2 + (y - center1[1]) ** 2 < radius1 ** 2
circle2 = (x - center2[0]) ** 2 + (y - center2[1]) ** 2 < radius2 ** 2
circle3 = (x - center3[0]) ** 2 + (y - center3[1]) ** 2 < radius3 ** 2
circle4 = (x - center4[0]) ** 2 + (y - center4[1]) ** 2 < radius4 ** 2

img = circle1 + circle2 + circle3 + circle4

# We use a mask that limits to the foreground: the problem that we are
# interested in here is not separating the objects from the background,
# but separating them one from the other.
mask = img.astype(bool)
img = img.astype(float)
img += 1 + 0.2 * np.random.randn(*img.shape)

# Convert the image into a graph with the value of the gradient on the
# edges.
# img = ndimage.gaussian_filter(img, sigma=(5, 5, 0), order=0)
graph = image.img_to_graph(img, mask=mask)

# Take a decreasing function of the gradient: we take it weakly
# dependent from the gradient the segmentation is close to a voronoi
graph.data = np.exp(-graph.data / graph.data.std())

print(graph)

A = graph
D = np.diag(np.ravel(np.sum(A,axis=1)))
L = D - A

# U has matrix of all eigenvectors arranged and l has sorted eigenvalues
l, U = la.eigh(L)

# Commented out code - Fiedler vector for 2-clustering alone
'''
f = U[:,1]
labels = np.ravel(np.sign(f))
k=2
'''

# Run K-Means on eigenvector matrix ( Other than 0th column )
# Input k here to specify required number of clusters
# means will have the list of K-means cluster centres
# labels show the different cluster labels detected : Note that some times
# K-Means doesn't converge and you might have 1 cluster lesser than 'K'
k = 4
means, labels = vq.kmeans2(U[:,1:k], k)
print(U.shape)
print(means)
print(labels.size)
print(np.unique(labels))

label_im = -np.ones(mask.shape)
label_im[mask] = labels

plt.matshow(img)
plt.matshow(label_im)
plt.show()
