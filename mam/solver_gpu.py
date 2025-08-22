import torch as T

def proj_simplex_scaled(W, q):
    # W: [B, R]; q: [B]
    # Sorting-based projection per row onto {x >= 0, sum x = q}
    B, R = W.shape
    W_sorted, _ = T.sort(W, dim=1, descending=True)
    css = T.cumsum(W_sorted, dim=1)
    r = T.arange(1, R+1, device=W.device, dtype=W.dtype).view(1, -1)
    tau = (css - q.view(-1,1)) / r
    cond = W_sorted > tau
    rho = cond.sum(dim=1).clamp(min=1) - 1         # index of last True per row
    tau_star = tau.gather(1, rho.view(-1,1)).squeeze(1)
    X = (W - tau_star.view(-1,1)).clamp_min(0)
    return X

def mam_gpu(Pi, D, q, maps, S_m, a_m, rho=1000.0, max_it=1000, tol=1e-4, dtype=T.float64, device="cuda"):
    """
    Pi:   [B,R]  initial transport plan rows (stacked columns π^m_:s)
    D:    [B,R]  cost columns D^m_:s aligned with Pi rows
    q:    [B]    column masses q^m_s
    maps: [B]    int64, row -> measure index m in [0..M-1]
    S_m:  [M]    number of columns per measure (S^m)
    a_m:  [M]    weights a_m = (1/S^m)/∑(1/S^j)
    """
    Pi = Pi.to(device=device, dtype=dtype).contiguous()
    D  = D.to(device=device, dtype=dtype).contiguous()
    q  = q.to(device=device, dtype=dtype)
    maps = maps.to(device=device, dtype=T.long)
    S_m = S_m.to(device=device, dtype=dtype)
    a_m = a_m.to(device=device, dtype=dtype)

    B, R = Pi.shape
    M = S_m.numel()

    ones = T.ones(R, dtype=dtype, device=device)

    def segmented_sum_rows(X):
        # X: [B,R]; returns per-measure sum over rows belonging to m: [M,R]
        out = T.zeros(M, R, dtype=dtype, device=device)
        out.index_add_(0, maps, X)  # sums rows into groups by 'maps'
        return out

    # Initialize p^m and p
    p_m = segmented_sum_rows(Pi)                      # sum over s of π^m_:s  → [M,R]
    p    = (a_m.view(-1,1) * p_m).sum(dim=0)          # weighted average over m → [R]

    for it in range(max_it):
        # ΔU_b per batch row: (p - p^m)/S^m
        p_m_per_row = p_m.index_select(0, maps)       # [B,R]
        Sm_per_row  = S_m.index_select(0, maps).view(-1,1)  # [B,1]
        deltaU = (p.view(1,-1) - p_m_per_row) / Sm_per_row  # [B,R]

        # Pre-projection point
        W = Pi + 2.0 * deltaU - D / rho               # [B,R]

        # Project each row onto Δ_R(q_b)
        Pi_new = proj_simplex_scaled(W, q) - deltaU   # [B,R]

        # Update p^m and p
        p_m = segmented_sum_rows(Pi_new)              # [M,R]
        p_new = (a_m.view(-1,1) * p_m).sum(dim=0)     # [R]

        # Stopping criterion (∞-norm on p)
        if T.max(T.abs(p_new - p)) < tol:
            Pi = Pi_new
            p = p_new
            break

        Pi = Pi_new
        p  = p_new

    return p, Pi

## Test 1 with centered digits
import pickle
N = 10
with open('../toy_examples/digit_datasets/3_centered.pkl', 'rb') as f:  # b_centers_MNIST #b_1_mat
    l_b = pickle.load(f)
b = l_b[:N]

mam_gpu(b, rho=.1,
    exact=False,
    computation_time=20, iterations_max=400, precision=10 ** -6,
    visualize=True, name=f'outputs_Centered.pkl')