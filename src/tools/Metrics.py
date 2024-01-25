import numpy as np
import torch
from scipy.spatial.distance import cdist

def frechet_distance(P, Q, metric='euclidean', clipping=False):
    """
    Compute discrete Fréchet distance between polygonal curves P and Q.

    P and Q are represented as matrices with each column corresponding to a point.
    # """
    if len(P[1]) != len(Q):
        print("P =", P, "Q =", Q)
        raise ValueError("Points in polygonal lines P and Q must have the same dimension.")
    # print(P)
    t = P[0].T
    y_true= P[1]
    d = 0
    # print(y_true[:,0].reshape(-1, 1).shape)
    for ii in range(5):
        # print(t.reshape(-1, 1).shape)
        true_P = torch.stack((t, y_true[:, ii]), dim=1)
        # print(true_P)
        pred_Q = torch.stack((t, Q[:, ii]), dim=1)

        # Compute pairwise distances, should not transpose
        couplings = cdist(true_P, pred_Q, metric= metric)
        # print("Couplings: ", couplings)
        # print("Couplings shape: ", couplings.shape)

        m, n = couplings.shape

        if clipping:
            # Clip the values in the coupling matrix at the average value of all distances
            avg =0
            for i in range(m):
                avg += np.mean(couplings[i, :])
            couplings = np.clip(couplings, a_min=None, a_max=avg/m)

        # Update couplings matrix; This loop is not needed because our pred and true data are always the same length.
        # for i in range(1, m):
        #     couplings[i, 0] = max(couplings[i-1, 0], couplings[i, 0])

        for j in range(1, n):
            couplings[j, 0] = max(couplings[j-1, 0], couplings[j, 0])
            couplings[0, j] = max(couplings[0, j-1], couplings[0, j])
            for i in range(1, m):
                carried_coupling = min(couplings[i-1, j-1], couplings[i, j-1], couplings[i-1, j])
                couplings[i, j] = max(carried_coupling, couplings[i, j])
        print('couple= ', couplings[i, j])
        d += couplings[m-1,n-1]
        print('d= ', d)
    return d/5


#calculates steady state error based on average of last n elements
def average_steady_state_error(y_true, y_pred, n=10):
    return torch.mean(torch.abs(y_true[-n:] - y_pred[-n:]))

if __name__ == '__main__':
    # Example usage
    # P = np.array([[1, 2, 3], [4, 5, 6]])  # Replace with your data
    # Q = np.array([[7, 8, 9, 10], [10, 11, 12, 13]])  # Replace with your data
    Q = np.arange(50).reshape(10, 5)
    a = np.arange(10).reshape(-1, 1)
    b = np.arange(50, 100).reshape(10, 5)
    P = [a,b]
    print(P[1])
    print(P)
    print(frechet_distance(P, Q))

    # print("Cosine Distance: ", frechet_distance(P, Q, metric='cosine')) # perhaps usefull for when we shift to the frequency domain
    # print("Canberra Distance: ", frechet_distance(P, Q, metric='canberra')) # Maybe usefull for near zero or zero magnitude signals
    # print("Chebyshev Distance: ", frechet_distance(P, Q, metric='chebyshev')) # takes the maximum deviation
    # print("Manhattan Distance: ", frechet_distance(P, Q, metric='cityblock')) # might be more usefull than euclidean when we want to
    # focus more on small deviations
    # def mean_frechet_distance(P, Q, width):
    #     "compartmentalizes the Frechet distance and takes the mean to give a more smooth result"
    #     return np.mean([frechet_distance(P[t: t + width], Q) for t in range(len(P)- width + 1)])
    #
    # np.random.seed(0)
    #
    # # Generate a base line for P and Q
    # base_line = np.linspace(0, 10, 20)
    #
    # # Create P and Q as similar lines with small random variations
    # P = np.array([base_line, base_line + np.random.uniform(-0.5, 0.5, 20)]).T
    # Q = np.array([base_line, base_line + np.random.uniform(-0.5, 0.5, 20)]).T
    # # P = np.array([base_line, base_line + np.random.uniform(-0.5, 0.5, 20)])
    # # Q = np.array([base_line, base_line + np.random.uniform(-0.5, 0.5, 20)])
    # print(P)
    # print(frechet_distance(P, Q))