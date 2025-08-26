import pandas as pd
import numpy as np

def extract_diagonal(matrix):
    """
    Extract the diagonal elements from a square matrix.
    
    Args:
        matrix: A 2D list representing a square matrix
        
    Returns:
        A list containing the diagonal elements
    """
    n = len(matrix)
    diagonal = []
    for i in range(n):
        if matrix[i][i] is not np.nan:
            diagonal.append(int(matrix[i][i]))
    return diagonal




def find_top_n_candidates(matrix, array, min_value_intersection=50, top_n=10):
    """
    Find multiple best candidates ranked by the criteria.
    Criteria:
    1. matrix[i][j] >= min_value_intersection and matrix[i][k] >= min_value_intersection [intersection ratios has to be at least min_value_intersection]
    2. Minimize array[i] + array[j] + array[k]]
    
    Args:
        matrix: 2D numpy array or list of lists
        array: 1D numpy array or list
        min_value: minimum preferred value for matrix elements
        top_n: number of top candidates to return
        
    Returns:
        list: List of tuples (i, j, k, array_sum, matrix_ij, matrix_ik)
    """
    matrix = np.array(matrix)
    array = np.array(array)
    
    n_rows, n_cols = matrix.shape
    n_array = len(array)
    max_index = min(n_rows, n_cols, n_array)
    
    candidates = []
    
    for i in range(max_index):
        for j in range(max_index):
            for k in range(max_index):
                # Skip if any indices are the same (i, j, k must all be different)
                if i == j or i == k or j == k:
                    continue
                
                matrix_ij = matrix[i, j]
                matrix_ik = matrix[i, k]
                array_sum = array[i] + array[j] + array[k]
                
                if matrix_ij >= min_value_intersection and matrix_ik >= min_value_intersection:
                    candidates.append((array_sum, {i, j, k}))

    # Sort by priority (1 first), then by array sum (ascending)
    candidates.sort(key=lambda x: (x[0]))
    # Remove duplicates (same sets of indices)
    seen = set()
    candidates = [c for c in candidates if frozenset(c[1]) not in seen and not seen.add(frozenset(c[1]))]

    # Return top candidates
    result = []
    for candidate in candidates[:top_n]:
        array_sum, ijkset = candidate
        ijkset = list(ijkset)
        result.append({'ijkset':ijkset, 
                       ijkset[0]: dataset_lenghts[ijkset[0]], 
                       ijkset[1]: dataset_lenghts[ijkset[1]], 
                       ijkset[2]: dataset_lenghts[ijkset[2]], ''
                       'array_sum': array_sum, 
                       'intersection_ratio '+str(ijkset[0])+ '-' +str(ijkset[1]): str(matrix[ijkset[0], ijkset[1]]),
                       'intersection_ratio '+str(ijkset[0])+ '-' +str(ijkset[2]): str(matrix[ijkset[0], ijkset[2]]),  
                       'intersection_ratio '+str(ijkset[1])+ '-' +str(ijkset[2]): str(matrix[ijkset[1], ijkset[2]])})

    return result

# Example usage
if __name__ == "__main__":
    df=pd.read_csv('/Users/sepide/Projects/amazon-reviews-analysis/Data/Intersection results/intersections.csv', header=None)
    dataset_lenghts = extract_diagonal(df.to_numpy())

    df_intersection_ratio = pd.read_csv('/Users/sepide/Projects/amazon-reviews-analysis/Data/Intersection results/Row_Normalized_Matrix_Percentage_.csv', header=None)
    intersection_ratio = df_intersection_ratio.to_numpy()
    intersection_ratio = intersection_ratio[1:, 1:].astype(np.float64)

    print(find_top_n_candidates(intersection_ratio, dataset_lenghts, min_value_intersection=50,top_n=10))
