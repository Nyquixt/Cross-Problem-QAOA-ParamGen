import numpy as np

def extract_Q(model, variables):
    """
    Extract the quadratic coefficient matrix Q from the model objective.
    """
    n = len(variables)
    Q = np.zeros((n, n))
    var_index_map = {var: i for i, var in enumerate(variables)}

    for (var1, var2), coeff in model.objective_expr.iter_quads():
        idx1 = var_index_map.get(var1)
        idx2 = var_index_map.get(var2)
        if idx1 is not None and idx2 is not None:
            Q[idx1, idx2] += coeff
            if idx1 != idx2:
                Q[idx2, idx1] += coeff  # ensure symmetry
        else:
            print(f"Warning: skipping non-variable quadratic term {str(var1)} * {str(var2)}")

    return Q

def extract_c(model, variables):
    """
    Extract the linear coefficient vector c from the model objective.
    """
    n = len(variables)
    c = np.zeros(n)
    var_index_map = {var: i for i, var in enumerate(variables)}

    for var, coeff in model.objective_expr.iter_terms():
        idx = var_index_map.get(var)
        if idx is not None:
            c[idx] = coeff

    return c

def extract_A_b(model, variables):
    """
    Return A_ineq, b_ineq, A_eq, b_eq for Ax<=b and Ax==b.
    Handles variables passed as dict {node: var} or list/iterable of vars.
    """
    # Normalize variables input
    if isinstance(variables, dict):
        nodes = sorted(variables.keys())
        var_list = [variables[i] for i in nodes]
    else:
        var_list = list(variables)
        nodes = list(range(len(var_list)))

    n = len(var_list)
    name2idx = {v.name: i for i, v in enumerate(var_list)}

    A_ineq, b_ineq, A_eq, b_eq = [], [], [], []

    for ct in model.iter_constraints():
        A_row = np.zeros(n, dtype=np.float64)

        # Aggregate coefficients on the same var (+= guards duplicates/self-loops)
        for var, coef in ct.lhs.iter_terms():
            A_row[name2idx[var.name]] += float(coef)

        rhs_value = float(ct.rhs.get_constant())
        sense = ct.sense.name  # 'LE', 'GE', or 'EQ'

        if sense == 'LE':
            A_ineq.append(A_row); b_ineq.append(rhs_value)
        elif sense == 'GE':
            A_ineq.append(-A_row); b_ineq.append(-rhs_value)
        elif sense == 'EQ':
            A_eq.append(A_row); b_eq.append(rhs_value)
        else:
            raise ValueError(f"Unknown sense: {sense}")

    A_ineq = np.vstack(A_ineq) if A_ineq else np.zeros((0, n), dtype=np.float64)
    b_ineq = np.asarray(b_ineq, dtype=np.float64) if b_ineq else np.zeros((0,), dtype=np.float64)
    A_eq   = np.vstack(A_eq)   if A_eq   else np.zeros((0, n), dtype=np.float64)
    b_eq   = np.asarray(b_eq,   dtype=np.float64) if b_eq else np.zeros((0,), dtype=np.float64)

    return A_ineq, b_ineq, A_eq, b_eq

def build_qubo(Q, c):
    Q_qubo = Q.copy()

    for i in range(c.shape[0]):
        Q_qubo[i, i] += c[i]
    
    return np.triu(Q_qubo) # take upper triangle

def extract_docplex_matrices(model):
    """
    Extract c, Q, A, b matrices from a docplex model.
    Returns:
        c (np.ndarray): linear coefficient vector
        Q (np.ndarray): quadratic coefficient matrix
        A (np.ndarray): constraint coefficient matrix
        b (np.ndarray): right-hand side vector
    """
    variables = list(model.iter_variables())
    Q = extract_Q(model, variables)
    c = extract_c(model, variables)
    A_ineq, b_ineq, A_eq, b_eq = extract_A_b(model, variables)
    return Q, c, A_ineq, b_ineq, A_eq, b_eq