import numpy as np
from scipy.spatial.distance import cdist

def frechet_distance(P, Q, metric='euclidean'):
    """
    Compute discrete Fréchet distance between polygonal curves P and Q.

    P and Q are represented as matrices with each column corresponding to a point.
    # """
    if P.shape[0] != Q.shape[0]:
        raise ValueError("Points in polygonal lines P and Q must have the same dimension.")

    # Compute pairwise distances
    couplings = cdist(P.T, Q.T, metric= metric)
    # print(couplings)
    #might not need to transpose
    # couplings = cdist(P, Q, metric= metric)


    m, n = couplings.shape

    # Update couplings matrix
    for i in range(1, m):
        couplings[i, 0] = max(couplings[i-1, 0], couplings[i, 0])

    for j in range(1, n):
        couplings[0, j] = max(couplings[0, j-1], couplings[0, j])
        for i in range(1, m):
            carried_coupling = min(couplings[i-1, j-1], couplings[i, j-1], couplings[i-1, j])
            couplings[i, j] = max(carried_coupling, couplings[i, j])

    return couplings[m-1, n-1]


if __name__ == '__main__':
    # Example usage
    P = np.array([[1, 2, 3], [4, 5, 6]])  # Replace with your data
    Q = np.array([[7, 8, 9, 10], [10, 11, 12, 13]])  # Replace with your data
    print(frechet_distance(P, Q))

    # print("Cosine Distance: ", frechet_distance(P, Q, metric='cosine')) # perhaps usefull for when we shift to the frequency domain
    # print("Canberra Distance: ", frechet_distance(P, Q, metric='canberra')) # Maybe usefull for near zero or zero magnitude signals
    # print("Chebyshev Distance: ", frechet_distance(P, Q, metric='chebyshev')) # takes the maximum deviation
    # print("Manhattan Distance: ", frechet_distance(P, Q, metric='cityblock')) # might be more usefull than euclidean when we want to
    # focus more on small deviations
    def mean_frechet_distance(P, Q, width):
        "compartmentalizes the Frechet distance and takes the mean to give a more smooth result"
        return np.mean([frechet_distance(P[t: t + width], Q) for t in range(len(P)- width + 1)])

    np.random.seed(0)

    # Generate a base line for P and Q
    base_line = np.linspace(0, 10, 20)

    # Create P and Q as similar lines with small random variations
    P = np.array([base_line, base_line + np.random.uniform(-0.5, 0.5, 20)]).T
    Q = np.array([base_line, base_line + np.random.uniform(-0.5, 0.5, 20)]).T
    # P = np.array([base_line, base_line + np.random.uniform(-0.5, 0.5, 20)])
    # Q = np.array([base_line, base_line + np.random.uniform(-0.5, 0.5, 20)])
    print(P)
    print(frechet_distance(P, Q))