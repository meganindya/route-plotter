import cv2
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from queue import PriorityQueue
import pickle

#G = ''
def main():
    #global G
    G = nx.read_gpickle("temp/graph.gpickle")
    nodes_count = len(G.nodes())
    start = nodes_count - 2
    end = nodes_count - 1
    prev = find_sh_path(G, start, end)
    path = [end]
    x = end
    while True:
        if prev[x] == None:    break
        path.insert(0, prev[x])
        x = prev[x]

    print(len(path))
    for i in path:    print(i, "\t", G.node[i]['pts'][0])

    """org = cv2.imread('../mazes/maze-2.jpg')
    org = cv2.resize(org, (400, 400), interpolation = cv2.INTER_AREA)
    b,g,r = cv2.split(org)
    org = cv2.merge([r,g,b])
    plt.imshow(org, cmap = 'gray')
    current = 0
    while True:
        n = G.node[path[current]]['pts'][0]
        #print(n[1], n[0])
        plt.plot(n[1], n[0], 'yo')
        if path[current] == end:   break
        pt = G[path[current]][path[current + 1]]['pts']
        plt.plot(pt[:,1], pt[:,0], 'm')
        current = current + 1
    plt.title('Shortest Path')
    plt.show()"""

    with open('temp/path', 'wb') as fp:
        pickle.dump(path, fp)

def heuristic(G, a, b):
       # Manhattan distance on a square grid
   a, b = G.node[a]['pts'][0], G.node[b]['pts'][0]
   return abs(a[1] - b[1]) + abs(a[0] - b[0])

def find_sh_path(G, start, goal):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0
    
    while not frontier.empty():
       current = frontier.get()
       if current == goal:      break
       for next in G.neighbors(current):
          new_cost = cost_so_far[current] + G[current][next]['weight']
          if next not in cost_so_far or new_cost < cost_so_far[next]:
             cost_so_far[next] = new_cost
             priority = new_cost + heuristic(G, goal, next)
             frontier.put(next, priority)
             came_from[next] = current
    return came_from

if __name__ == "__main__":
    main()
