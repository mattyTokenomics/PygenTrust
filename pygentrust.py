import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
import random
import tqdm

def initialize_local_trust_matrix(n=100):
    return np.zeros((n, n))

def initalize_global_trust_scores(n=100,p=10):
    if p > n or p <= 0:
        raise ValueError("p must be a positive integer less than or equal to n")

    # Initialize the first p elements with 1/p
    trust_scores = [1/p if i < p else 0 for i in range(n)]
    return trust_scores

def peer_interaction(local_scores, peer1, peer2, interaction):
    local_scores[peer1, peer2] += interaction
    return local_scores

def generate_random_peer_interactions(local_scores,m=1_000):
    num_peers = local_scores.shape[0]

    for _ in range(m):
        peer1, peer2 = random.sample(range(num_peers), 2)
        interaction = random.choice([-1, 1])
        local_scores = peer_interaction(local_scores, peer1, peer2, interaction)
    return local_scores

def normalize_local_scores(local_scores, global_scores):
    n = local_scores.shape[0]
    normalized_local_scores = lil_matrix((n, n))

    # Set diagonal elements to 0 to exclude self-voting
    local_scores.setdiag(0)

    for i in tqdm.tqdm(range(n)):
        row = local_scores.getrow(i)
        row_sum = row.data[row.data > 0].sum()

        if row_sum > 0:
            normalized_row = row.multiply(row > 0) / row_sum
            normalized_local_scores[i] = normalized_row
        else:
            # Handle the case where row sum is 0
            normalized_local_scores[i] = csr_matrix(global_scores)

    return normalized_local_scores.tocsr()

# From whitepaper, Algorithm 2: Basic EigenTrust algorithm
def basic_eigen_trust(local_scores, global_scores, alpha=0.1, convergence_threshold=0.01):
    # Ensure local_scores is a csr_matrix
    if not isinstance(local_scores, csr_matrix):
        local_scores = csr_matrix(local_scores) #If this step is failing try intializing local_scores as a coo_matrix (scipy.sparse)

    # Convert global_scores to a NumPy array if it is not already one
    if not isinstance(global_scores, np.ndarray):
        global_scores = np.array(global_scores)

    # Normalize the local_scores first
    local_scores = normalize_local_scores(local_scores, global_scores)

    while True:
        # Step 1: Multiply the transpose of normalized local scores with global scores
        new_global_scores = local_scores.T.dot(global_scores)

        # Step 2: Blend with existing global scores
        new_global_scores = (1 - alpha) * new_global_scores + alpha * global_scores

        # Step 3: Check for convergence
        delta = np.linalg.norm(new_global_scores - global_scores)
        if delta < convergence_threshold:
            break

        global_scores = new_global_scores

    return global_scores

def add_new_peers(local_scores, global_scores, new_peers_count, malicious_collective=False):
    n = local_scores.shape[0]
    new_size = n + new_peers_count
    new_local_scores = np.zeros((new_size, new_size))

    # Copy existing data
    new_local_scores[:n, :n] = local_scores

    # Initialize new entries in local_scores for malicious collective
    if (new_peers_count > 1) and malicious_collective:
        new_peer_score = 1 / (new_peers_count - 1)
        new_local_scores[n:, n:] = new_peer_score * np.ones((new_peers_count, new_peers_count))
        np.fill_diagonal(new_local_scores[n:, n:], 0)  # Set self-interaction to 0

    # Update global_scores for new peers
    new_global_scores = np.append(global_scores, np.zeros(new_peers_count))

    return new_local_scores, new_global_scores

if __name__ == "__main__":
    # Test your functions here
    n = 10
    p = 5
    m = 50
    alpha = 0.1
    convergence_threshold = 0.01

    local_scores = initialize_local_trust_matrix(n)
    local_scores = generate_random_peer_interactions(local_scores, m)
    initial_global_scores = initalize_global_trust_scores(n, p)
    normalized_local_scores = normalize_local_scores(local_scores, initial_global_scores)
    final_global_scores = basic_eigen_trust(normalized_local_scores, initial_global_scores, alpha, convergence_threshold)

    print(final_global_scores)
