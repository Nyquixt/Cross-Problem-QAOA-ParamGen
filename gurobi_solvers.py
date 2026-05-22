import gurobipy as gb
import os

from threading import local

def _get_gurobi_env():
    _tls = local()  # one env per worker thread/process
    env = getattr(_tls, "env", None)
    if env is None:
        env = gb.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.setParam("Threads", 1)      # avoid oversubscription
        env.start()
        _tls.env = env
    return env

def maxcut(env, graph, max_time=5, start=None):
    p = gb.Model(env=env)
    if max_time:
        p.setParam('TimeLimit', max_time)
    p.setParam('Threads', os.cpu_count())
    p.setParam('OutputFlag', 0)  # Suppress output

    # Variables
    vdict = {}
    for n in graph.nodes:
        vdict[n] = p.addVar(name='v_'+str(n), vtype=gb.GRB.BINARY)

    C_i = [vdict[i] + vdict[j] - 2 * vdict[i] * vdict[j] for i, j in graph.edges]
    p.setObjective(sum(C_i), gb.GRB.MAXIMIZE)
    # ---- MIP start (optional) ----
    if start is not None:
        if isinstance(start, dict):
            # start: {node: 0/1}
            for n, var in vdict.items():
                if n in start:
                    var.Start = float(start[n])  # must be numeric
        else:
            # start: list/tuple aligned with graph.nodes iteration order
            start = start >= 0.5
            start = start.tolist()
            nodes = list(graph.nodes)
            assert len(start) == len(nodes), "start list must match number of nodes"
            for n, val in zip(nodes, start):
                vdict[n].Start = float(val)
    p.optimize()
    return p.ObjVal # the optimal value

def mis(env, graph, max_time=5, start=None):
    p = gb.Model(env=env)
    if max_time:
        p.setParam('TimeLimit', max_time)
    p.setParam('Threads', os.cpu_count())
    p.setParam('OutputFlag', 0)  # Suppress output

    # Variables: one binary var per node (1 = selected in cover)
    vdict = {}
    for n in graph.nodes:
        vdict[n] = p.addVar(name=f'v_{n}', vtype=gb.GRB.BINARY)

    # Constraints: for every edge, at least one endpoint must be in the cover
    for u, v in graph.edges:
        p.addConstr(vdict[u] + vdict[v] <= 1, name=f'iset_{u}_{v}')

    # Objective: minimize the number of selected nodes
    p.setObjective(gb.quicksum(vdict[n] for n in graph.nodes), gb.GRB.MAXIMIZE)

    # ---- MIP start (optional) ----
    if start is not None:
        if isinstance(start, dict):
            # start: {node: 0/1}
            for n, var in vdict.items():
                if n in start:
                    var.Start = float(start[n])  # must be numeric
        else:
            # start: list/tuple aligned with graph.nodes iteration order
            # start = start >= 0.5
            start = start.clamp(0.0, 1.0).tolist()
            nodes = list(graph.nodes)
            assert len(start) == len(nodes), "start list must match number of nodes"
            for n, val in zip(nodes, start):
                vdict[n].Start = float(val)
    
    p.optimize()
    return p.ObjVal

def maxclique(env, graph, max_time=5, start=None):
    p = gb.Model(env=env)
    if max_time:
        p.setParam("TimeLimit", max_time)
    p.setParam("Threads", os.cpu_count())
    p.setParam("OutputFlag", 0)

    # Variables: one binary per node
    nodes = list(graph.nodes)
    x = {n: p.addVar(vtype=gb.GRB.BINARY, name=f"x_{n}") for n in nodes}

    # Build undirected edge set for quick lookup
    E = set()
    for u, v in graph.edges:
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        E.add((a, b))

    # Non-edge constraints: for every pair not in E, x_u + x_v <= 1
    for i in range(len(nodes)):
        u = nodes[i]
        for j in range(i + 1, len(nodes)):
            v = nodes[j]
            key = (u, v) if u < v else (v, u)
            if key not in E:
                p.addConstr(x[u] + x[v] <= 1, name=f"nonedge_{u}_{v}")

    # Objective: maximize clique size
    p.setObjective(gb.quicksum(x[n] for n in nodes), gb.GRB.MAXIMIZE)
    # ---- MIP start (optional) ----
    if start is not None:
        if isinstance(start, dict):
            # start: {node: 0/1}
            for n, var in x.items():
                if n in start:
                    var.Start = float(start[n])  # must be numeric
        else:
            # start: list/tuple aligned with graph.nodes iteration order
            # start = start >= 0.5
            start = start.clamp(0.0, 1.0).tolist()
            nodes = list(graph.nodes)
            assert len(start) == len(nodes), "start list must match number of nodes"
            for n, val in zip(nodes, start):
                x[n].Start = float(val)
    p.optimize()
    return p.ObjVal

def mvc(env, graph, max_time=5, start=None):
    p = gb.Model(env=env)
    if max_time:
        p.setParam('TimeLimit', max_time)
    p.setParam('Threads', os.cpu_count())
    p.setParam('OutputFlag', 0)  # Suppress output

    # Variables: one binary var per node (1 = selected in cover)
    vdict = {}
    for n in graph.nodes:
        vdict[n] = p.addVar(name=f'v_{n}', vtype=gb.GRB.BINARY)

    # Constraints: for every edge, at least one endpoint must be in the cover
    for u, v in graph.edges:
        p.addConstr(vdict[u] + vdict[v] >= 1, name=f'cover_{u}_{v}')

    # Objective: minimize the number of selected nodes
    p.setObjective(gb.quicksum(vdict[n] for n in graph.nodes), gb.GRB.MINIMIZE)

    # ---- MIP start (optional) ----
    if start is not None:
        if isinstance(start, dict):
            # start: {node: 0/1}
            for n, var in vdict.items():
                if n in start:
                    var.Start = float(start[n])  # must be numeric
        else:
            # start: list/tuple aligned with graph.nodes iteration order
            # start = start >= 0.5
            start = start.clamp(0.0, 1.0).tolist()
            nodes = list(graph.nodes)
            assert len(start) == len(nodes), "start list must match number of nodes"
            for n, val in zip(nodes, start):
                vdict[n].Start = float(val)
    p.optimize()
    return p.ObjVal