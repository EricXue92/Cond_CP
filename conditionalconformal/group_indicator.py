import numpy as np

def indicator_matrix(scalar_values, disc):
    scalar_values = np.array(scalar_values)

    # Create all possible intervals
    intervals = [(disc[i], disc[i + 1]) for i in range(len(disc) - 1)]

    # Initialize the indicator matrix
    matrix = np.zeros((len(scalar_values), len(intervals)))

    # Fill in the indicator matrix
    for i, value in enumerate(scalar_values):
        for j, (a, b) in enumerate(intervals):
            if a <= value < b:
                matrix[i, j] = 1

    return matrix
