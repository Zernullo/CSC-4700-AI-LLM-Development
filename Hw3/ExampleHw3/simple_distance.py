"""
Vector Distance Metrics Example

What this does:
    Calculates cosine angle, cosine similarity, and L1/L2 distances between
    two vectors, then prints the results for a simple example.

What you'll learn:
    - How to compute basic vector similarity and distance metrics
    - How to interpret the outputs for small vectors
"""

import numpy as np


def calculate_vector_metrics(v1, v2):
    """
    Calculate cosine angle, cosine similarity, L1 distance, and L2 distance
    between two 2-D vectors.
    
    Args:
        v1: First vector (list or array-like)
        v2: Second vector (list or array-like)
    
    Returns:
        Dictionary with metrics: theta, cosine_similarity, l1_distance, l2_distance
    """
    v1 = np.array(v1)
    v2 = np.array(v2)
    
    # Cosine similarity
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cosine_similarity = dot_product / (norm_v1 * norm_v2)
    
    # Cosine angle theta (in radians)
    theta = np.arccos(np.clip(cosine_similarity, -1, 1))
    
    # L1 distance (Manhattan distance)
    l1_distance = np.linalg.norm(v1 - v2, ord=1)
    
    # L2 distance (Euclidean distance)
    l2_distance = np.linalg.norm(v1 - v2, ord=2)
    
    return {
        "theta": theta,
        "cosine_similarity": cosine_similarity,
        "l1_distance": l1_distance,
        "l2_distance": l2_distance
    }

# Example usage
if __name__ == "__main__":
    warm_vector = [0.9, 1.1]
    hot_vector = [2.1, 1.8]
    
    results = calculate_vector_metrics(warm_vector, hot_vector)
    print(f"Cosine angle theta: {results['theta']:.4f} radians ({np.degrees(results['theta']):.2f} degrees)")
    print(f"Cosine similarity: {results['cosine_similarity']:.4f}")
    print(f"L1 distance: {results['l1_distance']:.4f}")
    print(f"L2 distance: {results['l2_distance']:.4f}")