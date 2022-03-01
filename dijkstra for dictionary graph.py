import numpy as np

graph = {
    'd00': {'d10': 2, 'd01': 2},
    'd01': {'d02': 4, 'd00': 1},
    'd02': {'d12': 2, 'd01': 2},
    'd04': {'d14': 1, 'd05': 1},
    'd05': {'d15': 2, 'd04': 3},
    'd10': {'d20': 3, 'd00': 1},
    'd12': {'d22': 1, 'd02': 4},
    'd14': {'d24': 3, 'd04': 3, 'd15': 2},
    'd15': {'d25': 2, 'd05': 1, 'd14': 1},
    'd20': {'d10': 2, 'd21': 2},
    'd21': {'d22': 1, 'd20': 3},
    'd22': {'d32': 2, 'd12': 2, 'd23': 4, 'd21': 2},
    'd23': {'d33': 1, 'd24': 3, 'd22': 1},
    'd24': {'d14': 1, 'd25': 2, 'd23': 4},
    'd25': {'d35': 1, 'd15': 2, 'd24': 3},
    'd31': {'d41': 1, 'd21': 2, 'd32': 2},
    'd32': {'d42': 1, 'd22': 1, 'd33': 1},
    'd33': {'d23': 4, 'd32': 2},
    'd35': {'d45': 1, 'd25': 2},
    'd40': {'d50': 1, 'd41': 1},
    'd41': {'d31': 1, 'd42': 1, 'd40': 2},
    'd42': {'d52': 4, 'd32': 2, 'd41': 1},
    'd45': {'d55': 1, 'd35': 1},
    'd50': {'d40': 2},
    'd52': {'d42': 1, 'd53': 2},
    'd53': {'d54': 2, 'd52': 4},
    'd54': {'d55': 1, 'd53': 2},
    'd55': {'d45': 1, 'd54': 2}
}

def path_dijkstra(graph,start,goal):
    shortest_distance = {}
    track_predecessor = {}
    unseenNodes = graph
    infinity = 999999
    track_path = []

    for node in unseenNodes:
        shortest_distance[node] = infinity
    shortest_distance[start] = 0

    while unseenNodes:

        min_distance_node = None

        for node in unseenNodes:
            if min_distance_node is None:
                min_distance_node = node
            elif shortest_distance[node] < shortest_distance[min_distance_node]:
                min_distance_node = node

        path_options = graph[min_distance_node].items()

        for child_node, weight in path_options:

            if weight + shortest_distance[min_distance_node] < shortest_distance[child_node]:
                shortest_distance[child_node] = weight + shortest_distance[min_distance_node]
                track_predecessor[child_node] = min_distance_node

        unseenNodes.pop(min_distance_node)

    currentNode = goal

    while currentNode != start:
        try:
            track_path.insert(0,currentNode)
            currentNode = track_predecessor[currentNode]
        except KeyError:
            print("Path is not reachable")
            break
    if shortest_distance[goal] != infinity:
        print("Shortest distance is " + str(shortest_distance[goal]))
        print("Optimal Path is " + str(track_path))

path_dijkstra(graph,'d55','d00')
