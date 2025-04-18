def depth_first_search(graph):
    for vertex in graph:
        vertex.visited = False
        vertex.previous = None

    for vertex in graph:
        if not vertex.visited:
            recursive_dfs_helper(vertex)


def recursive_dfs_helper(vertex):
    vertex.visited = True
    for neighbor in vertex.neighbors:
        if not neighbor.visited:
            neighbor.previous = vertex
            recursive_dfs_helper(neighbor)


def iterative_dfs(graph, start):
    for vertex in graph:
        vertex.visited = False
        vertex.previous = None

    stack = [start]

    while stack:
        current = stack.pop()
        if not current.visited:
            current.visited = True
            for neighbor in current.neighbors:
                if not neighbor.visited:
                    neighbor.previous = current
                    stack.append(neighbor)