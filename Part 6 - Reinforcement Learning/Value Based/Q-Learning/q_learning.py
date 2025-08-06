import numpy as np

# Setting the parameters
gamma = 0.75 # Discount factor
alpha = 0.1  # Learning rate

# Defining the environment
location_to_state = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4,
    'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9,
    'K': 10, 'L': 11
}
actions = list(range(12))  
R = np.array([
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]])

# Building the AI solution
# Making a mapping from the states to the locations
state_to_location = {state: location for location, state in location_to_state.items()}

def route(start, end):
    R_new = np.copy(R)
    ending_state = location_to_state[end]
    R_new[ending_state, ending_state] = 1000
    Q = np.array(np.zeros([12, 12]))
    for _ in range(1000):
        current_state = np.random.randint(0, 12)
        playable_actions = np.nonzero(R_new[current_state, :])[0].tolist()
        next_state = np.random.choice(playable_actions)
        TD = R_new[current_state, next_state] + gamma * Q[next_state, np.argmax(Q[next_state,])] - Q[current_state, next_state]
        Q[current_state, next_state] += alpha * TD

    path = [start]
    next_loc = start

    while next_loc != end:
        starting_state = location_to_state[next_loc]
        next_state = np.argmax(Q[starting_state, :])
        next_loc = state_to_location[next_state]
        path.append(next_loc)
    return path

def best_route(start, inter, end):
    return route(start, inter) + route(inter, end)[1:]

print(best_route('E', 'K', 'G'))