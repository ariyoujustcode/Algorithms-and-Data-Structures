from collections import deque

def breadth_first_search(graph, source):
    for vertex in graph:
        vertex.distance = float("inf")
        vertex.previous = None

    source.distance = 0
    source.previous = None
    bfs_queue = deque([source])

    while bfs_queue:  # or while !bfs_queue.isEmpty()
        current_vertex = bfs_queue.popleft()

        for neighbor in current_vertex.neighbors():
            if neighbor.distance == float("inf"):
                neighbor.distance = current_vertex.distance + 1
                neighbor.previous = current_vertex
                bfs_queue.append(neighbor)

def print_path(graph, source, vertex):
    if vertex == source:
        print(source, end="")

    elif vertex.previous == None:
        print("No path from source to vertex.")

    else:
        print_path(graph, source, vertex.previous)
        print(" ->", vertex, end="")