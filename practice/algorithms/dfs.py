def depth_first_search(graph, source):
    for vertex in graph:
        vertex.distance = float('inf')
        vertex.previous = None

    source.distance = 0
    source.previous = None
    stack = [source]