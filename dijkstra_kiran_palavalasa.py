import numpy as np
import time
import cv2
from queue import PriorityQueue

# Define the move functions
move_up = lambda node: ((node[0] - 1, node[1]), 1)
move_down = lambda node: ((node[0] + 1, node[1]), 1)
move_left = lambda node: ((node[0], node[1] - 1), 1)
move_right = lambda node: ((node[0], node[1] + 1), 1)
move_up_left = lambda node: ((node[0] - 1, node[1] - 1), np.sqrt(2))
move_up_right = lambda node: ((node[0] - 1, node[1] + 1), np.sqrt(2))
move_down_left = lambda node: ((node[0] + 1, node[1] - 1), np.sqrt(2))
move_down_right = lambda node: ((node[0] + 1, node[1] + 1), np.sqrt(2))

# Define obstacles
def rectangle1(x, y):
    return 0 <= y <= 100 and 100 <= x <= 150

def rectangle2(x, y):
    return 150 <= y <= 250 and 100 <= x <= 150

def hexagon(x, y):
    return (75/2) * abs(x-300)/75 + 50 <= y <= 250 - (75/2) * abs(x-300)/75 - 50 and 225 <= x <= 375

def triangle(x, y):
    return (200/100) * (x-460) + 25 <= y <= (-200/100) * (x-460) + 225 and 460 <= x <= 510

eqns = {
    "Rectangle1": rectangle1,
    "Rectangle2": rectangle2,
    "Hexagon": hexagon,
    "Triangle": triangle
}


map_width, map_height, clearance = 600, 250, 5
pixels = np.full((map_height, map_width, 3), 255, dtype=np.uint8)

# Fill pixels based on obstacles and clearance
for i in range(map_height):
    for j in range(map_width):
        for eqn in eqns.values():
            if eqn(j, i):
                pixels[i, j] = [0, 0, 0]  # obstacle
                break
        else:
            for eqn in eqns.values():
                if eqn(j, i-clearance) or eqn(j, i+clearance) or eqn(j-clearance, i) or eqn(j+clearance, i):
                    pixels[i, j] = [192, 192, 192]  

def is_valid_node(node):
    x, y = node
    return 0 <= x < map_width and 0 <= y < map_height and (pixels[y, x] == [255, 255, 255]).all()

def is_goal(current_node, goal_node):
    return current_node == goal_node

def backtrack_path(parents, start_node, goal_node, animation):
    height, _, _ = animation.shape
    path, current_node = [goal_node], goal_node
    while current_node != start_node:
        path.append(current_node)
        current_node = parents[current_node]
        animation[height - 1 - current_node[1], current_node[0]] = (0, 255, 0)  # Mark path (in green)
        cv2.imshow('Animation', animation)
        cv2.waitKey(1)
    path.append(start_node)
    return path[::-1]

def dijkstra(start_node, goal_node):
    open_list = PriorityQueue()
    closed_list = set()
    cost_to_come = {start_node: 0}
    cost = {start_node: 0}
    parent = {start_node: None}
    open_list.put((0, start_node))
    visited = set([start_node])
    height, _, _ = pixels.shape
    animation = pixels.copy()

    while not open_list.empty():
        _, current_node = open_list.get()

        # Mark current node as visited (in blue)
        mark_visited(animation, current_node)

        if is_goal(current_node, goal_node):
            path = backtrack_path(parent, start_node, goal_node, animation)
            # Mark start and goal nodes as red
            mark_start_goal(animation, start_node, goal_node)
            # Show the final animation
            show_animation(animation)
            # Print the final cost
            print(f"Final Cost: {cost[goal_node]}")
            return path

        # Expand the current node
        expand_node(current_node, cost_to_come, cost, parent, open_list, closed_list, visited)

        # Show the animation after expanding the current node
        show_animation(animation)

    # If we reach this point, there is no path from start to goal
    # Show the final animation
    show_animation(animation)
    print("No path found.")
    return None


def expand_node(current_node, cost_to_come, cost, parent, open_list, closed_list, visited):
    for move_func in [move_up, move_down, move_left, move_right, move_up_left, move_up_right, move_down_left, move_down_right]:
        new_node, move_cost = move_func(current_node)

        if is_valid_node(new_node) and new_node not in closed_list:
            new_cost_to_come = cost_to_come[current_node] + move_cost

            if new_node not in cost_to_come or new_cost_to_come < cost_to_come[new_node]:
                cost_to_come[new_node] = new_cost_to_come
                cost[new_node] = new_cost_to_come
                parent[new_node] = current_node
                open_list.put((new_cost_to_come, new_node))
                visited.add(new_node)
                # Mark new node as visited (in green)
                mark_visited(pixels, new_node, (0, 255, 0))


def show_animation(animation):
    cv2.imshow('Animation', animation)
    cv2.waitKey(1)


def mark_visited(animation, node, color=(255, 0, 0)):
    height, _, _ = animation.shape
    animation[height - 1 - node[1], node[0]] = color


def mark_start_goal(animation, start_node, goal_node):
    height, _, _ = animation.shape
    animation[height - 1 - start_node[1], start_node[0]] = (0, 0, 255)
    animation[height - 1 - goal_node[1], goal_node[0]] = (0, 0, 255)

# Get valid start and goal nodes from user input
while True:
    # Get start and goal nodes from user input
    start_str = input("\nEnter the start node (in the format 'x y'): ")
    start_node = tuple(map(int, start_str.split()))
    goal_str = input("Enter the goal node (in the format 'x y'): ")
    goal_node = tuple(map(int, goal_str.split()))

    # Check if start and goal nodes are valid
    if not is_valid_node(start_node):
        print(f"Error: Start node {start_str} is not valid. Please input a valid node.")
        continue
    if not is_valid_node(goal_node):
        print(f"Error: Goal node {goal_str} is not valid. Please input a valid node.")
        continue
    # If both nodes are valid, break out of the loop
    break
if cv2.waitKey(1) & 0xFF == ord('q'):
    cv2.destroyAllWindows()


start_time = time.time()
path = dijkstra(start_node, goal_node)
if path is None:
    print("\nError: No path found.")
else:
    print("\nGoal Node Reached!")
end_time = time.time()
print("Runtime:", end_time - start_time, "seconds\n")
