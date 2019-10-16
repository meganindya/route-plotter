import cv2
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
import math
from math import sqrt
from PIL import Image
from skimage.morphology import skeletonize
from skimage.util import invert
import sknw


def main():
    img = cv2.imread("temp/solidified.jpg")
    img = cv2.resize(img, (400, 400), interpolation = cv2.INTER_AREA)
    img = invert(img)
    #cv2.imshow("dfd", img)
    org = cv2.imread('../mazes/maze-2.jpg')
    org = cv2.resize(org, (400, 400), interpolation = cv2.INTER_AREA)
    skel = make_skeleton(img)
    graph = make_graph(skel, img, org)
    plot_graph(graph, org)
    
def make_skeleton(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #BınaryThreshold + OtsuThreshold + BinaryThreshold
    retval, threshold = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    retval, threshold2 = cv2.threshold(threshold, 10, 255, cv2.THRESH_BINARY_INV)
    threshold2[threshold2 == 255] = 1
        #Skeletonize the Thresholded Image
    return skeletonize(threshold2)

def make_graph(skel, image, org):
    graph = sknw.build_sknw(skel, multi=False)

    gr_sensitivity = 15;
    gr_lb = np.array([60 - gr_sensitivity, 100, 50])
    gr_ub = np.array([60 + gr_sensitivity, 255, 255])

    rd_sensitivity = 15;
    rd_lb = np.array([0 - rd_sensitivity, 50, 50])
    rd_ub = np.array([10 + rd_sensitivity, 255, 255])
    gr = color_coordinate(org, gr_lb, gr_ub)
    rd = color_coordinate(org, rd_lb, rd_ub)
    #print(gr, "    ", rd)

    add_node_to_graph(graph, [gr[1], gr[0]])
    add_node_to_graph(graph, [rd[1], rd[0]])
    add_edge_to_graph(graph, [gr[1], gr[0]], 2)
    add_edge_to_graph(graph, [rd[1], rd[0]], 1)
    return graph

def plot_graph(graph, image):
    nx.write_gpickle(nx.Graph(graph), 'temp/graph.gpickle')
    """b,g,r = cv2.split(image)
    image = cv2.merge([r,g,b])
    plt.imshow(image, cmap = 'gray')
    
        #Draw Edges by 'pts'
    for (s,e) in graph.edges():
        #print(s, "\t", e, "\t", graph[s][e]['weight'])
        ps = graph[s][e]['pts']
        plt.plot(ps[:,1], ps[:,0], 'c')

        #Draw Node by 'o'
    node, nodes = graph.node, graph.nodes()
    ps = np.array([node[i]['o'] for i in nodes])
    #print(ps)
    plt.plot(ps[:,1], ps[:,0], 'w.')

    plt.title('Track Graph')
    #plt.savefig('Overlay_Maze.jpg')
    plt.show()"""

def add_node_to_graph(graph, cd):
    l = len(graph.nodes())
    in_int = np.array([cd], dtype = 'int16')
    in_float = np.array([float(cd[0]), float(cd[1])])
    graph.add_node(l, pts = in_int, o = in_float)

def add_edge_to_graph(graph, cd, n):
    node, nodes = graph.node, graph.nodes()
    node_no = -1; min_dist = math.inf
    for i in range(len(nodes) - 2):
        dist = distance(node[i]['pts'][0], cd)
        if dist < min_dist:
            min_dist = dist
            node_no = i
    ex_node = len(nodes) - n
    p1 = (node[node_no]['pts'][0][0], node[node_no]['pts'][0][1])
    p2 = (node[ex_node]['pts'][0][0], node[ex_node]['pts'][0][1])
    ar = points_between(p1, p2)
    graph.add_edge(node_no, ex_node, pts = np.array(ar, dtype = 'int16'), weight = float(len(ar)))

def points_between(p1, p2):
    xs = range(p1[0] + 1, p2[0]) or [p1[0]]
    ys = range(p1[1] + 1, p2[1]) or [p1[1]]
    return np.array([[x,y] for x in xs for y in ys])

def distance(a, b):
    return sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)

def color_coordinate(image, lb, ub):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lb, ub)
    imge, contours, hier = cv2.findContours(mask.copy(), 1, 2)

    moments = [cv2.moments(cnt) for cnt in contours]
    centroids = [(int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])) for M in moments]
    return centroids[0]

if __name__ == "__main__":
    main()
