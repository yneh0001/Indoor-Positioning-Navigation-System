
import random
import tracemalloc
import time
from collections import deque
from operator import add, truediv
from platform import node
from tempfile import TemporaryDirectory

from fibonacci_heap import FibonacciHeap
INFINITY = float("inf")

class C:
    def __init__(self,x,y,n=0):
        self.x = x
        self.y = y
        self.n = n
        
    def set_n(self, value):
        self.n = value

def initialiser():
    matrix = []
    for x in range(26):
        matrix.append([])    
        for y in range (15,42):
            matrix[x].append(-1)
    return matrix


def location(matrix):
 # initialise point 
    location = [C(0,41),C(0,34),C(0,28),C(0,15),C(4,36),C(14,40),C(7,28),C(14,28),
                C(8,26),C(6,20),C(14,20),C(25,41),C(25,26),C(25,23),C(25,17),C(25,15)]
    for i in range(len(location)):
        matrix[location[i].x][location[i].y-15] = location[i]
    return matrix

def pathway(matrix):
    #initialise pathway
    #H1
    for x in range (1,4):
        for y in range(15,42):
            matrix[x][y-15] = C(x,y)

    #H2
    for x in range (4,22):
        for y in range(41,42):
            matrix[x][y-15] = C(x,y)

    #H3
    for x in range (16,25):
        for y in range(15,42):
            matrix[x][y-15] = C(x,y)

    #H4
    for x in range (8,16):
        for y in range(15,20):
            matrix[x][y-15] = C(x,y)

    #H5
    for x in range (4,7):
        for y in range(17,20):
            matrix[x][y-15] = C(x,y)

    #H6
    for x in range (4,8):
        for y in range(24,28):
            matrix[x][y-15] = C(x,y)
    # #H7        
    # for x in range (9,14):
    #     for y in range(26,28):
    #         matrix[x][y-15] = C(x,y)
    for x in range (7,21):
        matrix[x][5] = C(x,20)

    for x in range (9,15):
        for y in range (26,28):
            matrix[x][y-15]=C(x,y)
    
    return matrix
def traffic(z,matrix):
    print("\nTraffic created: ")
    for i in range (z):
        x = random.randint(0,25)
        y = random.randint(0,26)
        matrix[x][y] = -1
        print((x,y),end = " ")
    return matrix

def addEdgeToGraph(adj,matrix):
    
    for a in range (len(matrix)):
        for b in range (len(matrix)+1):
            cur = matrix[a][b] 
            if cur != -1: 
                if a+1 != len(matrix):
                    next = matrix[a+1][b]
                    if next != -1:
                        add_edge(adj,cur,next)
                
                if b+1 != len(matrix)+1:
                    next = matrix[a][b+1]
                    if next != -1:
                        add_edge(adj,cur,next)

                if(a  == (len(matrix)-1)) and b+1 != len(matrix) + 1: 
                    next = matrix[a][b+1]
                    if next != -1:
                        add_edge(adj,cur,next)

                if b  == len(matrix) and a+1 != len(matrix):
                    next = matrix[a+1][b]
                    if next != -1:
                        add_edge(adj,cur,next)
    return                     
def add_edge(adj, src, dest):

    adj[src.n].append(dest);
    adj[dest.n].append(src);


def BFS(adj, src, dest, v, pred, dist):

    queue = []
    visited = [False for i in range(v)];

    for i in range(v):

        dist[i] = 1000000
        pred[i] = -1;
    

    visited[src.n] = True;
    dist[src.n] = 0;
    queue.append(src);

    # standard BFS algorithm
    while (len(queue) != 0):
        u = queue[0];
        queue.pop(0);
        # print(adj[u])
        for i in range(len(adj[u.n])):
            # print(adj)
            temp = u.n
            # print(temp)
            if (visited[adj[temp][i].n] == False):
                visited[adj[temp][i].n] = True;
                dist[adj[temp][i].n] = dist[temp] + 1;
                pred[adj[temp][i].n] = u;
                queue.append(adj[temp][i]);

                # We stop BFS when we find
                # destination.
                if (adj[u.n][i] == dest):
                    return True;

    return False;

# utility function to print the shortest distance
# between source vertex and destination vertex
def printShortestDistance(adj, src,dest, v):
    
    # predecessor[i] array stores predecessor of
    # i and distance array stores distance of i
    # from s

    pred=[0 for i in range(v)]
    dist=[0 for i in range(v)];
    
    if (BFS(adj, src,dest, v, pred, dist) == False):
        print("Path does not exist !")
        
        

    # vector path stores the shortest path
    path = []
    crawl = dest.n;
    path.append(dest);
    while (pred[crawl] != -1):
        # print(crawl)
        path.append(pred[crawl]);
        crawl = pred[crawl].n;
    
    # distance from source is in distance array
    
    if dist[dest.n] != 1000000:
        print("Shortest path length is : " + str(dist[dest.n]), end = '')

        # printing path from source to destination
        print("\nPath is :  ")
        
        for i in range(len(path)-1, -1, -1):
            print((path[i].x,path[i].y), end = " ")

def count(matrix):
    count = 0 
    for a in range (len(matrix)):
        for b in range (len(matrix)+1):
            if matrix[a][b] == -1:
                continue
            else:
                s = C(a,b)
                matrix[a][b].set_n(count)
                count += 1
    # print(matrix[0][13].n)
    return matrix        


    
    # no. of vertices



class Graph:
    
    def addEdgeDijsktra (self,matrix):
        graph_edges = []
        for a in range (len(matrix)):
            for b in range (len(matrix)+1):
                cur = matrix[a][b] 
                if cur != -1: 
                    if a+1 != len(matrix):
                        next = matrix[a+1][b]
                        if next != -1:
                            graph_edges.append((cur,next,1))
                            graph_edges.append((next,cur,1))
                    
                    if b+1 != len(matrix)+1:
                        next = matrix[a][b+1]
                        if next != -1:
                            graph_edges.append((cur,next,1))
                            graph_edges.append((next,cur,1))

                    if(a  == (len(matrix)-1)) and b+1 != len(matrix) + 1: 
                        next = matrix[a][b+1]
                        if next != -1:
                            graph_edges.append((cur,next,1))
                            graph_edges.append((next,cur,1))

                    if b  == len(matrix) and a+1 != len(matrix):
                        next = matrix[a+1][b]
                        if next != -1:
                            graph_edges.append((cur,next,1))
                            graph_edges.append((next,cur,1))

        self.nodes = set()
        
        for edge in graph_edges:
            self.nodes.update([edge[0], edge[1]])
        
        
           
        self.adjacency_list = {node: set() for node in self.nodes}
        for edge in graph_edges:
            self.adjacency_list[edge[0]].add((edge[1], edge[2]))
        return self.adjacency_list

    def shortest_path(self, start_node, end_node):
        """Uses Dijkstra's algorithm to determine the shortest path from
        start_node to end_node. Returns (path, distance).
        """
        unvisited_nodes = self.nodes.copy()  # All nodes are initially unvisited.
        # print(unvisited_nodes)
        # Create a dictionary of each node's distance from start_node. We will
        # update each node's distance whenever we find a shorter path.
        distance_from_start = {
            node: (0 if node == start_node else INFINITY) for node in self.nodes
        }

        # Initialize previous_node, the dictionary that maps each node to the
        # node it was visited from when the the shortest path to it was found.
        previous_node = {node: None for node in self.nodes}
        
        while unvisited_nodes:
            # Set current_node to the unvisited node with shortest distance
            # calculated so far.
            current_node = min(
                unvisited_nodes, key=lambda node: distance_from_start[node]
            )
            unvisited_nodes.remove(current_node)
            
            # If current_node's distance is INFINITY, the remaining unvisited
            # nodes are not connected to start_node, so we're done.
            if distance_from_start[current_node] == INFINITY:
                
                break

            # For each neighbor of current_node, check whether the total distance
            # to the neighbor via current_node is shorter than the distance we
            # currently have for that node. If it is, update the neighbor's values
            # for distance_from_start and previous_node.
            for neighbor, distance in self.adjacency_list[current_node]:
                new_path = distance_from_start[current_node] + distance
                if new_path < distance_from_start[neighbor]:
                    distance_from_start[neighbor] = new_path
                    previous_node[neighbor] = current_node
                    # print(current_node.x,current_node.y)
            if current_node == end_node:
                
                break # we've visited the destination node, so we're done
            
        # To build the path to be returned, we iterate through the nodes from
        # end_node back to start_node. Note the use of a deque, which can
        # appendleft with O(1) performance.
        path = deque()
        current_node = end_node
        
        while previous_node[current_node] is not None:
            
            path.appendleft(current_node)
            current_node = previous_node[current_node]
            # print(current_node)
            
        path.appendleft(start_node)
        if distance_from_start[end_node] == INFINITY:
            print("Path does not exist !")
        else: 
            print("Shortest path length is : " + str(distance_from_start[end_node]), end = '')
            print("\nPath is :  ")
            for i in range (len(path)):
                print((path[i].x,path[i].y), end=' ')
        # print(path)
        
        
        return path, distance_from_start[end_node]


def addEdgeAStar (matrix):
    graph_edges = []
    for a in range (len(matrix)):
        for b in range (len(matrix)+1):
            cur = matrix[a][b] 
            if cur != -1: 
                if a+1 != len(matrix):
                    next = matrix[a+1][b]
                    if next != -1:
                        graph_edges.append((cur,next,1))
                        graph_edges.append((next,cur,1))
                
                if b+1 != len(matrix)+1:
                    next = matrix[a][b+1]
                    if next != -1:
                        graph_edges.append((cur,next,1))
                        graph_edges.append((next,cur,1))

                if(a  == (len(matrix)-1)) and b+1 != len(matrix) + 1: 
                    next = matrix[a][b+1]
                    if next != -1:
                        graph_edges.append((cur,next,1))
                        graph_edges.append((next,cur,1))

                if b  == len(matrix) and a+1 != len(matrix):
                    next = matrix[a+1][b]
                    if next != -1:
                        graph_edges.append((cur,next,1))
                        graph_edges.append((next,cur,1))

    nodes = set()
    
    for edge in graph_edges:
        nodes.update([edge[0], edge[1]])

    adjacency_list = {node: set() for node in nodes}
    for edge in graph_edges:
        adjacency_list[edge[0]].add((edge[1], edge[2]))
       
    return adjacency_list

class AStarGraph:

    def __init__(self, adjacency_list):
        self.adjacency_list = adjacency_list

    def get_neighbors(self, v):
        return self.adjacency_list[v]

    # heuristic function with equal values for all nodes
    def h(self, n):
        H = {
            'A': 1,
            'B': 1,
            'C': 1,
            'D': 1
        }

        return 1
        # return H[n]

    def a_star_algorithm(self, start_node, stop_node):
        open_list = set([start_node])
        closed_list = set([])

        # g contains current distances from start_node to all other nodes
        # the default value (if it's not found in the map) is +infinity
        g = {}

        g[start_node] = 0

        # parents contains an adjacency map of all nodes
        parents = {}
        parents[start_node] = start_node

        while len(open_list) > 0:
            n = None

            # find a node with the lowest value of f() - evaluation function
            for v in open_list:
                if n == None or g[v] + self.h(v) < g[n] + self.h(n):
                    n = v;

            if n == None:
                print('Path does not exist!')
                return None

            if n == stop_node:
                reconst_path = []

                while parents[n] != n:
                    reconst_path.append(n)
                    n = parents[n]

                reconst_path.append(start_node)

                reconst_path.reverse()
                if (len(reconst_path)-1) != 0:
                    print("Shortest path length is :",len(reconst_path)-1)
                    print("Path is : ")
                    for i in range(len(reconst_path)):
                        reconst_path[i] = (reconst_path[i].x,reconst_path[i].y)
                        print(reconst_path[i], end = " ")
                        
        
            
                return reconst_path

            # for all neighbors of the current node do
            for (m, weight) in self.get_neighbors(n):
                if m not in open_list and m not in closed_list:
                    open_list.add(m)
                    parents[m] = n
                    g[m] = g[n] + weight

                else:
                    if g[m] > g[n] + weight:
                        g[m] = g[n] + weight
                        parents[m] = n

                        if m in closed_list:
                            closed_list.remove(m)
                            open_list.add(m)
            open_list.remove(n)
            closed_list.add(n)

        print('Path does not exist!')
        return None

def tracing_start():
    tracemalloc.stop()
    print("resetTracing Status : ", tracemalloc.is_tracing())
    tracemalloc.start()
    print("Tracing Status : ", tracemalloc.is_tracing())
def tracing_mem():
    first_size, first_peak = tracemalloc.get_traced_memory()
    peak = first_peak/(1024*1024)
    print("Peak Size in MB : ", peak)      

def BFSdriver (matrix,start,end):
    v = 450
    adj = [[] for i in range(v)]
    addEdgeToGraph(adj,matrix)
    tracing_start()
    startTime = time.time()
    printShortestDistance(adj, start,end, v)
    endTime = time.time()
    print("\ntime elapsed {} milli seconds".format((endTime-startTime)*1000))
    tracing_mem()
    return 

def Dijdriver (matrix,start,end):
    g = Graph()
    g.addEdgeDijsktra(matrix)
    tracing_start()
    startTime = time.time()
    g.shortest_path(start,end)
    endTime = time.time()
    print("\ntime elapsed {} milli seconds".format((endTime-startTime)*1000))
    tracing_mem()
    return

def AStardriver(matrix,start,end):
    adj = addEdgeAStar(matrix)
    tracing_start()
    startTime = time.time()
    g = AStarGraph(adj)
    g.a_star_algorithm(start,end)
    endTime = time.time()
    print("\ntime elapsed {} milli seconds".format((endTime-startTime)*1000))
    tracing_mem()
    return 

def mainAlgo(matrix,x1,x2,y1,y2):
    
    start = matrix[x1][y1]
    end = matrix[x2][y2]
    print("\n")
    print("\nBFS")
    try:
        BFSdriver(matrix,start,end)
    except:
        print("Invalid set of coordinates")
        pass
    print("\n")
    print("Dijkstra")
    try:
        Dijdriver(matrix,start,end)
    except: 
        print("Invalid set of coordinates")
        pass
    print("\n")
    
    print("A Star")
    try:
        AStardriver(matrix,start,end)
    except:
        print("Invalid set of coordinates")
    print("\n")
    return

def driver():
    while True:
        print("1. Navigation")
        print("2. Quit")
        while True:
            try:
                userInput = int(input("Enter your option: "))
            except:
                print("Invalid option")
            else:
                break
        if userInput == 1:
            
            matrix = initialiser()
            matrix = pathway(matrix)
            while True:
                try:
                    x1 = int(input("Enter first x coordinate: "))
                    if x1 < 0 or x1 > 25:
                        raise Exception
                except:
                    print("Invalid Coordinate")
                    continue
                else:
                    break
            while True:
                try:
                    y1 = int(input("Enter first y coordinate: "))
                    y1 = y1 -15
                    if y1 < 0 or y1 > 26:
                        raise Exception
                except:
                    print("Invalid Coordinate")
                    continue
                else:
                    break

            while True:
                try:
                    x2 = int(input("Enter second x coordinate: "))
                    if x2 < 0 or x2 > 25:
                        raise Exception
                except:
                    print("Invalid Coordinate")
                    continue
                else:
                    break

            while True:
                try:
                    y2 = int(input("Enter second y coordinate: "))
                    y2 = y2 - 15
                    if y2 < 0 or y2 > 26:
                        raise Exception
                except:
                    print("Invalid Coordinate")
                    continue
                else:
                    break
            
            while True:
                print("Press -1 to continue with the obstacles created")
                while True:

                    try:
                        z1 = int(input("Enter X coordinate: "))
                        if z1 == -1:
                            break
                        if z1 < 0 or z1 > 25:
                            raise Exception
                    except:
                        print("Invalid Coordinate")
                        continue
                    else:
                        break
                if z1 == -1:
                    break            
                while True:
                    try:
                        z2 = int(input("Enter Y coordinate: "))
                        if z2 == 100:
                            break
                        z2 = z2- 15
                        
                        if z2 < 0 or z2 > 26:
                            raise Exception
                    except:
                        print("Invalid Coordinate")
                        continue
                    else:
                        break
                if z2 == 100:
                    break         
                matrix[z1][z2] = -1 
            matrix = location(matrix)
            matrix = count(matrix)
            # for i in range (len(matrix)):
            #     for j in range(len(matrix)):
            #         print((i,j),matrix[i][j])     
            mainAlgo(matrix,x1,x2,y1,y2)
        if userInput == 2: 
            break

        
        

driver()
