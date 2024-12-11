import numpy as np

# Define the states and observations
states = ['L', 'NL']
observations = ['NC', 'NC', 'C', 'C', 'NC']  # Observations for 5 days

# Initial probabilities
pi = {'L': 0.5, 'NL': 0.5}

# Transition probabilities
transitions = {
    'L': {'L': 0.6, 'NL': 0.4},
    'NL': {'L': 0.3, 'NL': 0.7}
}

# Emission probabilities
emissions = {
    'L': {'C': 0.85, 'NC': 0.15},
    'NL': {'C': 0.15, 'NC': 0.85}
}

# Number of days
num_days = len(observations)

# Initialize the Viterbi table
viterbi_table = np.zeros((len(states), num_days))

# Initialize the first column of the Viterbi table
for i, state in enumerate(states):
    viterbi_table[i, 0] = pi[state] * emissions[state][observations[0]]

# Populate the Viterbi table
for t in range(1, num_days):
    for j, current_state in enumerate(states):
        max_prob = 0
        for i, previous_state in enumerate(states):
            prob = viterbi_table[i, t-1] * transitions[previous_state][current_state] * emissions[current_state][observations[t]]
            if prob > max_prob:
                max_prob = prob
        viterbi_table[j, t] = max_prob

# Print the Viterbi table
print("Viterbi Table:")
print("Day\tL\tNL")
for t in range(num_days):
    print(f"{t+1}\t{viterbi_table[0, t]:.10f}\t{viterbi_table[1, t]:.10f}")

# Optionally, you can also return the most likely states
most_likely_states = []
for t in range(num_days):
    if viterbi_table[0, t] > viterbi_table[1, t]:
        most_likely_states.append('L')
    else:
        most_likely_states.append('NL')

print("\nMost Likely States:")
print(most_likely_states)