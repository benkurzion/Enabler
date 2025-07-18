import numpy as np
import math
from tkinter import filedialog
from PIL import Image
import matplotlib.pyplot as plt


def open_image_file() -> np.array:
    '''
    Prompts user for image file and returns either
    - 3D numpy array with RGB values
    - None
    '''
    file_path = filedialog.askopenfilename(
        title="Select an Image File",
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.ico")]
    )

    if file_path:
        print(f"Selected image file: {file_path}")
        try:
            img = Image.open(file_path)
            print(f"Image opened: {img.format}, size: {img.size}")
            new_size_scaled = (128, 128)
            img_downsized_scaled = img.resize(new_size_scaled)
            pix_downsized_scaled = np.array(img_downsized_scaled.getdata()).reshape(img_downsized_scaled.size[0], img_downsized_scaled.size[1], 3)
            return pix_downsized_scaled
        except Exception as e:
            print(f"Error opening image: {e}")
    else:
        print("No image file selected.")

    return None


def similarity_absolute(p1 : np.ndarray, p2 : np.ndarray) -> int:
    '''
    A larger similarity is assigned for similar pixels. 
    Maximum value = 1.0
    '''
    if p1.shape[0] != p2.shape[0]:
        raise Exception("Something is wrong. Not given an RGB array")
    if np.linalg.norm(p1-p2) == 0:
        return 1
    return 1 / np.linalg.norm(p1-p2)



class Node:
    '''
    A node in a graph network. Each pixel in an image is represented as a node.

    Parameters
    ----------

    rgb : np.ndarray
        An array containing the red, green, and blue intensities at this pixel

    i : int
        The coordinate at which this pixel resides in the image along the vertical

    j : int
        The coordinate at which this pixel resides in the image along the horizontal

    edges : list
        A list of all of the outgoing edges from this node
    '''
    def __init__(self, rgb : np.ndarray, i : int, j : int, edges : list):
        self.rgb = rgb
        self.i = i
        self.j = j
        self.edges = edges


class Edge:
    '''
    A directed edge in a weighted graph network

    Parameters
    ----------

    node_to : list
        A list [i,j] denoting the coordinates of sink node

    node_from : list
        A list [i,j] denoting the coordinates of source node

    similarity : int
        A similarity measure between *node_to* and *node_from*

    weight_override : bool
        An optional flag indicating that the edge weight is hard coded in as 1
    '''

    def __init__(self, node_to: list, node_from: list, similarity : int = 0, weight_override: bool = False):
        self.node_to = node_to
        self.node_from = node_from
        if weight_override:
            self.weight = np.inf # source and sink should have infinite capacity
        else:
            self.weight = similarity

class Graph: 
    '''
    A parameterized graph for a min-cut algorithm

    Parameters
    ----------

    img : np.ndarray
        A 3D array with RGB values representing an image

    depth : int
        Determines how pixels are connected in the graph. *depth = 1* indicates connections between directly adjacent pixels (cardinal directions & diagonals). 
        *depth = 2* indicates additional connections to the layer of pixels, 2 hops away from a node.

    '''

    def __init__(self, img : np.ndarray, depth : int):
        height = img.shape[0]
        width = img.shape[1]
        
        self.img = img
        self.height = height
        self.width = width
        if depth < 1:
            raise ValueError("Depth must have a non-negative value")
        else:
            self.depth = depth

        self.construct_graph()
    
    def construct_graph(self):
        '''
        The graph is represented as a 2-D array of nodes. Each node contains a list of outgoing edges. 
        '''
        self.graph = np.zeros(shape=(self.height + 1, self.width), dtype=Node)
        for i in range(self.height):
            for j in range(self.width):
                self.graph[i][j] = Node(rgb=self.img[i][j], i=i, j=j, edges=None)
    
        # Now all of the nodes are set. Connect with edges
        for i in range(self.height):
            for j in range(self.width):
                self.graph[i][j].edges = self.get_edges(i, j)

        # Add in the source and sink nodes
        # Source is connected to top left
        # Sink is connected to bottom right
        source = Node(rgb=None, i=self.height, j=0, edges=[Edge(node_from=[self.height, 0], node_to=[0,0], weight_override=True)])
        sink = Node(rgb=None, i=self.height, j=1, edges=None)
        self.graph[self.height - 1][self.width - 1].edges.append(Edge(node_from=[self.height-1, self.width-1], node_to=[self.height, 1], weight_override=True))
        self.graph[self.height][0] = source
        self.graph[self.height][1] = sink

    def get_edges(self, i, j) -> list:
        '''
        Returns all of the edges from a node
        '''
        edges = []
        for d_j in range (self.depth + 1):
            if j + d_j < self.width:
                for d_i in range (1, self.depth + 1):
                    if i + d_i < self.height:
                        edges.append(Edge(node_from=[i, j], node_to=[i + d_i, j + d_j], similarity=similarity_absolute(self.graph[i][j].rgb, self.graph[i + d_i][j + d_j].rgb)))
                    if i - d_i >= 0:
                        edges.append(Edge(node_from=[i, j], node_to=[i - d_i, j + d_j], similarity=similarity_absolute(self.graph[i][j].rgb, self.graph[i - d_i][j + d_j].rgb)))
                if d_j > 0:
                    edges.append(Edge(node_from=[i, j], node_to=[i, j + d_j], similarity=similarity_absolute(self.graph[i][j].rgb, self.graph[i][j + d_j].rgb)))
            if j - d_j >= 0 and d_j != 0:
                for d_i in range (1, self.depth + 1):
                    if i + d_i < self.height:
                        edges.append(Edge(node_from=[i, j], node_to=[i + d_i, j - d_j], similarity=similarity_absolute(self.graph[i][j].rgb, self.graph[i + d_i][j - d_j].rgb)))
                    if i - d_i >= 0:
                        edges.append(Edge(node_from=[i, j], node_to=[i - d_i, j - d_j], similarity=similarity_absolute(self.graph[i][j].rgb, self.graph[i - d_i][j - d_j].rgb)))
                if d_j > 0: 
                    edges.append(Edge(node_from=[i, j], node_to=[i, j - d_j], similarity=similarity_absolute(self.graph[i][j].rgb, self.graph[i][j - d_j].rgb)))
        return edges



def ford_fulkerson_algo(graph : Graph) :
    '''
    Performs the min_cut / max_flow algorithm on the graph to separate distinct objects
    '''

    source = graph.graph[graph.height][0]
    # Continue finding paths from source to sink and update the capacities of edges used
    flag = True
    while flag:
        flag = False
        parent = bfs(graph=graph, source=source)
        if parent is None:
            print("No more paths. Ford Fulkerson terminated")
        else:
            flag = True
            # Find the maximum flow possible to push through this path
            max_flow = np.inf
            path_node = [graph.height, 1]
            while path_node[0] != graph.height or path_node[1] != 0:
                edge = parent[path_node[0]][path_node[1]][2]
                max_flow = min(max_flow, edge.weight)
                path_node = [parent[path_node[0]][path_node[1]][0], parent[path_node[0]][path_node[1]][1]]
            # Update the edge weights along this path with the max_flow
            path_node = [graph.height, 1]
            while path_node[0] != graph.height or path_node[1] != 0:
                edge = parent[path_node[0]][path_node[1]][2]
                edge.weight = edge.weight - max_flow
                path_node = [parent[path_node[0]][path_node[1]][0], parent[path_node[0]][path_node[1]][1]]

    resulting_img = np.zeros(shape=(graph.height, graph.width, 3))
    queue = [[0, 0]]
    visited = np.zeros(shape=(graph.height, graph.width), dtype=bool)
    while queue:
        node = queue.pop()
        node = graph.graph[node[0]][node[1]]
        for edge in node.edges:
            neighbor = edge.node_to
            # Check if neighbor is sink node
            if neighbor[0] == graph.height and neighbor[1] == 1:
                pass
            elif not visited[neighbor[0]][neighbor[1]] and edge.weight > 0:
                visited[neighbor[0]][neighbor[1]] = True
                queue.append([neighbor[0], neighbor[1]])
                neighbor_node = graph.graph[neighbor[0]][neighbor[1]]
                resulting_img[neighbor[0]][neighbor[1]][:] = neighbor_node.rgb
    
    # Display the image
    plt.imshow(resulting_img.astype(np.uint8))
    plt.axis('off')
    plt.show()




def bfs(graph : Graph, source : Node) -> list:
    '''
    Performs a breadth first search of the graph from source node *s* to sink node *t*. Returns parent array if there is a valid path.
    '''
    queue = [[source.i, source.j]]
    visited = np.zeros(shape=(graph.height + 1, graph.width), dtype=bool)
    parent = np.zeros(shape=(graph.height + 1, graph.width), dtype=list)
    while queue:
        node = queue.pop()
        node = graph.graph[node[0]][node[1]]
        for edge in node.edges:
            neighbor = edge.node_to
            # Check if neighbor is sink node
            if neighbor[0] == graph.height and neighbor[1] == 1:
                parent[neighbor[0]][neighbor[1]] = [node.i, node.j, edge]
                return parent
            if not visited[neighbor[0]][neighbor[1]] and edge.weight > 0:
                visited[neighbor[0]][neighbor[1]] = True
                parent[neighbor[0]][neighbor[1]] = [node.i, node.j, edge]
                queue.append([neighbor[0], neighbor[1]])
    return None



# Run
img_rgb = open_image_file()
graph = Graph(img=img_rgb, depth=1)
ford_fulkerson_algo(graph=graph)
