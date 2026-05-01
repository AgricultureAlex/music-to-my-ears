""" THIS CODE USES SCIENTIFIC PITCH NOTATION """

from collections import defaultdict

def build_debruijn_graph(kgrams):
    graph = defaultdict(list) #graph[node] = [neighbors]
    in_degree = defaultdict(int) #in_degree[node] = number of incoming edges
    out_degree = defaultdict(int) #out_degree[node] = number of outgoing edges

    for kgram in kgrams: #breaks it into prefix and suffix and puts them in the graph with directed edge from prefix to suffix (potentially overlapping)
        notes = kgram.split()
        prefix = tuple(notes[:-1])
        suffix = tuple(notes[1:])
        graph[prefix].append(suffix)

        out_degree[prefix] += 1
        in_degree[suffix] += 1

    return graph, in_degree, out_degree


def find_start_node(graph, in_degree, out_degree):
    """ need a good starting node that has out_degree - in_degree = 1, if not (ie. there's a cycle) then any node with out_degree > 0 will work """
    start = None
    for node in graph:
        if out_degree[node] -    in_degree[node] == 1:
            return node
        if out_degree[node] > 0:
            start = node
    return start


def eulerian_path(graph, start):
    stack = [start]
    path = []

    while stack:
        node = stack[-1] # look at current node (top of stack)
        if graph[node]: # if it has an outgoing edge, follow it and remove it from the graph
            next_node = graph[node].pop()
            stack.append(next_node)
        else:
            path.append(stack.pop()) #if no edges are left, we're done with this node and we add it to the final path then backtrack

    return path[::-1] #need it reveresed bc we add nodes to the path when we backtrack, so the path is built in reverse order


def reconstruct_string(path): #nodes overlap by k-1 characters so when we print it out, we should only print the new character each node adds
    result = list(path[0])
    for node in path[1:]:
        result.append(node[-1])
    return " ".join(result)


def shortest_sequence_from_kgrams(kgrams): #this is our pipeline basically
    graph, in_degree, out_degree = build_debruijn_graph(kgrams) #define structures
    start = find_start_node(graph, in_degree, out_degree) #find starting point
    path = eulerian_path(graph, start) #build path
    return reconstruct_string(path) #return the path as a string

kgrams = [
    "C4 E4 G4",
    "E4 G4 B4",
    "G4 B4 D5",
    "B4 D5 G5"
]

print(shortest_sequence_from_kgrams(kgrams))
