from scipy.sparse import csc_matrix, kron, hstack, vstack
import scipy
from scipy import optimize
import time
import numpy as np

def exact_wasserstein_barycenter(b, M_dist):
    """
    Inputs:
    *b: (n x M_) M_ probability measures, each measure has a support of dimension n
    *M_dist: (n x m) matrix giving distances between the locations of supports
    
    Outputs:
    *p_: (n x 1) barycenric probability measure
    *Distance_exact: (float) Wasserstein distance between p_ and the probability measures stored in b
    *Pi_exact: (n x m) Transport matrix between p_ and the M_ probability measures b[0] to b[M_-1]
    *Time: (float) time spent to compute the result
    
    Infos:
    LP resolution solving: Wasserstein_barycenter(b) (all b[i]'s have the same support)
    *function to minimize: c @ x /M_
    *constraints: A @ x = p_ and B @ x = b
    The massive linear program is run with the HiGHS method (see more on HiGHS.dev)
    """
    # Time management:
    start = time.time()
    
    # dimension parameters
    M_ = len(b)
    resolution = len(b[0])
    R = resolution
    S = np.array([resolution] * M_)
    sumS = np.sum(S)

    # Ponderated distance matrix as a vector c.T (1,R_*S_)
    D = np.tile(M_dist, M_)
    c = np.reshape(D, (R*sumS,)) / M_

    # Building A: Constraint (right sum of Pi to get p)
    shape_A = ((M_-1), sumS)
    # indices values
    indices_values_A = np.append(np.ones(S[0]), - np.ones(S[1]))
    # lines indices:
    line_indices_A = np.linspace(0,S[0]+S[1] - 1, S[0]+S[1])
    # column indices:
    column_indices_A = np.zeros(S[0]+S[1])
    index = 0
    for m in range(1,M_-1):
        indices_values_A = np.append(indices_values_A, np.append(np.ones(S[m]), - np.ones(S[m+1])) )
        index = index + S[m - 1]
        line_indices_A = np.append(line_indices_A, np.linspace(0,S[m]+S[m+1] - 1, S[m]+S[m+1]) + index )
        column_indices_A = np.append(column_indices_A, np.ones(S[m]+S[m+1]) * m)
    # Building one small matrix
    A = csc_matrix((indices_values_A, (column_indices_A, line_indices_A)), shape_A)
    # Using Kronecker product to get the total matrix
    A = kron(np.eye(R), A)

    # # Building B: Constraint (left sum of Pi to get q)
    # from scipy.sparse import hstack
    # B = csc_matrix(np.eye(sumS))
    # B = hstack([B]*R)

    # Building A_eq = concatenate((A, B), axis=0)
    A_eq = A # vstack([A,B])

    # b_eq:
    b_eq = np.zeros(R*(M_-1)) # np.append(np.zeros(R*(M_-1)),b)
    
    # Resolution:
    res = scipy.optimize.linprog(c, A_eq=A_eq, b_eq=b_eq, method='highs')
    
    # computing time:
    end = time.time()
    Time = np.round((end-start), 2)
    
    # Output
    # Distance:
    Distance_exact = res.fun
    # Transport matrix:
    Pi_exact = res.x
    Pi_exact = np.reshape(Pi_exact, (R, sumS))
    # Optimal barycentric probability distribution:
    p_ = np.zeros(R)
    for r in range(R):
        p_[r] = sum( Pi_exact[r,S[0]:S[0]+S[1]] ) 

    return(p_, Distance_exact, Pi_exact, Time)