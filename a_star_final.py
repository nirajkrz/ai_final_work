import heapq

# Define the directed graph as an adjacency list with costs
graph = {
    "E1": {"E2": 2},
    "E2": {"E3": 3},
    "E3": {"E4": 4, "M3": 10},
    "E4": {"E1": 5},
    "M3": {"M4": 2, "S4": 30},
    "M4": {"M5": 3},
    "M5": {"M1": 4, "J2": 10},
    "M1": {"M2": 1},
    "M2": {"M3": 2},
    "J2": {"J3": 4},
    "J3": {"S4": 10, "J1": 5},
    "J1": {"J2": 3},
    "S4": {"S1": 3},
    "S1": {"S2": 2},
    "S2": {"S3": 2},
    "S3": {"S4": 3},
}

# Define heuristic values
heuristics = {
    "E1": 30, "E2": 30, "E3": 30, "E4": 30,
    "M1": 20, "M2": 20, "M3": 20, "M4": 20, "M5": 20,
    "J1": 10, "J2": 10, "J3": 10,
    "S1": 0, "S2": 0, "S3": 0, "S4": 0,
}

def a_star(graph, heuristics, start, goal):
    # Priority queue
    open_list = []
    heapq.heappush(open_list, (0, start))  # (f(n), node)

    # Tracking costs and parents
    g_costs = {start: 0}
    parents = {start: None}

    while open_list:
        # Get the node with the lowest f(n)
        _, current = heapq.heappop(open_list)

        # Goal check
        if current == goal:
            # Reconstruct path
            path = []
            while current:
                path.append(current)
                current = parents[current]
            return path[::-1], g_costs[goal]

        # Explore neighbors
        for neighbor, cost in graph.get(current, {}).items():
            tentative_g = g_costs[current] + cost
            if neighbor not in g_costs or tentative_g < g_costs[neighbor]:
                g_costs[neighbor] = tentative_g
                f_cost = tentative_g + heuristics[neighbor]
                heapq.heappush(open_list, (f_cost, neighbor))
                parents[neighbor] = current

    return None, float("inf")  # No path found

# Run A* algorithm
start = "E1"
goal = "S3"
path, cost = a_star(graph, heuristics, start, goal)

# Output results
print("Optimal Path:", " -> ".join(path))
print("Total Cost:", cost)
