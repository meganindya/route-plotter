import pickle
import math
from math import sqrt
import networkx as nx
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import invert
import os

G = ''
def main():
    img = cv2.imread('../mazes/maze-2.jpg')
    img = cv2.resize(img, (400, 400), interpolation = cv2.INTER_AREA)
    with open('temp/path', 'rb') as fp:
        path = pickle.load(fp)
    global G
    G = nx.read_gpickle("temp/graph.gpickle")
    #ims = load_images('dtest')

    b_lb = np.array([110,50,50])
    b_ub = np.array([130,255,255])

    y_lb = np.array([20, 100, 100])
    y_ub = np.array([30, 255, 255])

    c_b = c_center(img, b_lb, b_ub)
    c_y = c_center(img, y_lb, y_ub)
    length = dist(c_b, c_y) * 2 

    end = len(G.nodes()) - 1
    path_r = path_red(length, end, path)
    for i in path_r:    print(i)

    draw_map(img, end, path, path_r, c_b, c_y)

def n_node(n):
    return G.node[n]['pts'][0]

def dist(a, b):
    return sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)

def angle(a, b, c):
    m1 = slope(a, b)
    m2 = slope(b, c)
    a1 = math.degrees(math.atan((m2 - m1) / (1 + (m1 * m2))))
    return a1
    
def slope(a, b):
    if (b[1] - a[1]) == 0:  return 2**500
    else:   return (b[0] - a[0]) / (b[1] - a[1])

def path_red(length, end, path, i = 0):
    path_r = path.copy()
    while True:
        if i == (len(path_r) - 2):    break
        if path_r[i] == -1:     i = i + 1; continue
        x = i + 1
        while True:
            if path_r[x] == end:   break
            distance = dist(n_node(path_r[i]), n_node(path_r[x]))
            if distance < length:
                path_r[x] = -1
                x = x + 1
                continue
            ang = abs(angle(n_node(path_r[i]), n_node(path_r[x]), n_node(path_r[x + 1])))
            if ang > 2:   break
            path_r[x] = -1; x = x + 1
        i = i + 1
    return list(filter(lambda a: a != -1, path_r))

def draw_map(img, end, path, path_r, db, dy):
    b,g,r = cv2.split(img)
    img = cv2.merge([r,g,b])
    plt.imshow(img, cmap = 'gray')
    current = 0
    while True:
        if path[current] == end:   break
        pt = G[path[current]][path[current + 1]]['pts']
        plt.plot(pt[:,1], pt[:,0], 'y')
        current = current + 1
    for x in path_r:
        plt.plot(n_node(x)[1], n_node(x)[0], 'wo')
    plt.title('Shortest Path')
    plt.plot(db[0], db[1], 'y.')
    plt.plot(dy[0], dy[1], 'b.')
    plt.savefig('temp/path.jpg')
    plt.show()

def c_center(image, lb, ub):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lb, ub)
    imge, contours, hier = cv2.findContours(mask.copy(), 1, 2)

    moments = [cv2.moments(cnt) for cnt in contours]
    centroids = [(int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])) for M in moments]
    return centroids[0]


if __name__ == "__main__":
    main()
