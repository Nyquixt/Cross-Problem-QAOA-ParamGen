# Quantum Machine Learning
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Standard Python libraries
import numpy as np
import random
import os
import pickle
import argparse
from copy import deepcopy

from gurobi_solvers import _get_gurobi_env, maxcut, mis, maxclique, mvc
from rnn import LSTMNet, ConditionedLSTM
from dataset import generate_graphs, generate_graphs
from qaoa import QAOAMaxcutCost, QAOAMaxCutShot, QAOAMISCost, QAOAMISShot, QAOAMaxCliqueCost, QAOAMaxCliqueShot, QAOAMVCCost, QAOAMVCShot
from utils import test_step, test_step_g2v, mis_empirical_stats_from_samples, maxcut_empirical_stats_from_samples, maxclique_empirical_stats_from_samples, mvc_empirical_stats_from_samples

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset", type=str, default="toy")
    parser.add_argument("--problem", type=str, default="maxcut")
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--test-steps", type=int, default=0)
    parser.add_argument("--use-g2v", action="store_true")
    parser.add_argument("--use-uni", action="store_true") # unihetco single
    parser.add_argument("--normalize", action="store_true") # normalize expval
    parser.add_argument("--seed", type=int, default=32)

    args = parser.parse_args()

    dataset = args.dataset
    n_layers = args.p
    problem = args.problem
    epochs = args.epochs
    batch_size = args.batch_size
    use_g2v = args.use_g2v
    use_uni = args.use_uni
    seed = args.seed
    horizon = args.horizon
    normalize = args.normalize
    test_steps = args.test_steps

    print("=============================================")
    print("dataset", dataset)
    print("n_layers", n_layers)
    print("problem", problem)
    print("epochs", epochs)
    print("batch_size", batch_size)
    print("use_g2v", use_g2v)
    print("use_uni", use_uni)
    print("horizon", horizon)
    print("normalize", normalize)
    print("seed", seed)

    test_path = f'./dataset_{args.dataset}/testset/graphs.pkl'
    if use_g2v:
        test_embedding_path = f'./dataset_{args.dataset}/testset/embeddings.npy'
    else:
        test_embedding_path = f'./dataset_{args.dataset}/testset/{problem}_unihetco_embeddings.pkl'

    # Fix the seed for reproducibility, which affects all random functions in this demo
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using MPS device: {device}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {device}")
    else:
        device = torch.device("cpu")
        print(f"Using CPU device: {device}")

    # Generate data

    if os.path.exists(test_path):
        print("Found test graphs pkl, Loading...")
        with open(test_path, 'rb') as infile:
            test_graphs = pickle.load(infile)
    else:
        test_graphs = generate_graphs(n_graphs=20, n_nodes_range=[12, 12], seed=50)
        with open(test_path, 'wb') as outfile:
            pickle.dump(test_graphs, outfile)
    
    print("Getting optimal solutions...")
    test_solutions = []
    for graph in test_graphs:
        env = _get_gurobi_env()
        env.setParam('OutputFlag', 0)
        env.start()
        solver_args = {
            "env": env,
            "graph": graph,
            "max_time": 100.0
        }
        if problem == 'maxcut':
            solution_val = maxcut(**solver_args)
        elif problem == 'mis':
            solution_val = mis(**solver_args)
        elif problem == 'maxclique':
            solution_val = maxclique(**solver_args)
        elif problem == 'mvc':
            solution_val = mvc(**solver_args)
        env.close()
        test_solutions.append(solution_val)

    if problem == 'maxcut':
        test_graph_cost_list = [QAOAMaxcutCost(g, n_layers=n_layers, normalized=normalize) for g in test_graphs]
        test_graph_shots_list = [QAOAMaxCutShot(g, n_layers=n_layers, n_shots=5000) for g in test_graphs]

    elif problem == 'mis':
        test_graph_cost_list = [QAOAMISCost(g, n_layers=n_layers, normalized=normalize) for g in test_graphs]
        test_graph_shots_list = [QAOAMISShot(g, n_layers=n_layers, n_shots=5000) for g in test_graphs]

    elif problem == 'maxclique':
        test_graph_cost_list = [QAOAMaxCliqueCost(g, n_layers=n_layers, normalized=normalize) for g in test_graphs]
        test_graph_shots_list = [QAOAMaxCliqueShot(g, n_layers=n_layers, n_shots=5000) for g in test_graphs]

    elif problem == 'mvc':
        test_graph_cost_list = [QAOAMVCCost(g, n_layers=n_layers, normalized=normalize) for g in test_graphs]
        test_graph_shots_list = [QAOAMVCShot(g, n_layers=n_layers, n_shots=5000) for g in test_graphs]

    if use_g2v or use_uni:
        if use_uni:
            with open(test_embedding_path, "rb") as infile:
                test_embeddings = pickle.load(infile)
        else:
            test_embeddings = np.load(test_embedding_path)
        lstm = ConditionedLSTM(p=n_layers, device=device, graph_dim=test_embeddings.shape[-1], T=horizon)
        test_embeddings = [row.copy() for row in test_embeddings]  # each is a (D,) np array
    else:
        lstm = LSTMNet(p=n_layers, device=device, T=horizon)
    state_dict = torch.load(f"./models_{args.dataset}/{problem}_p={n_layers}_T={horizon}_epochs={epochs}_bs={batch_size}_g2v={use_g2v}_uni={use_uni}_normalize={normalize}_seed={seed}.pth", map_location=device)
    lstm.load_state_dict(state_dict)
    lstm = lstm.to(device)

    approximation_ratio_list = []
    percent_optimal_list = []
    all_feasible_ratios = []
    for i, (graph_cost, graph_shots) in enumerate(zip(test_graph_cost_list, test_graph_shots_list)): # iterate through the dataset
        model = deepcopy(lstm) # reset to meta-model
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        # Perform a single testing step
        if use_g2v or use_uni:
            params, _ = test_step_g2v(graph_cost, test_embeddings[i], model, optimizer, test_steps)  # returns a PyTorch scalar tensor
        else:
            params, _ = test_step(graph_cost, model, optimizer, test_steps)
        
        # Calculate AR
        _params = params[-1].view(2, n_layers)
        
        bitstrings = graph_shots(_params).detach().cpu().numpy()
        
        if problem == "maxcut":
            exp_val, feasible_ratio, percent_optimal = maxcut_empirical_stats_from_samples(bitstrings, test_graphs[i], optimal_solution=test_solutions[i])
        
        if problem == 'mis':
            exp_val, feasible_ratio, percent_optimal = mis_empirical_stats_from_samples(bitstrings, test_graphs[i], optimal_solution=test_solutions[i])

        if problem == 'maxclique':
            exp_val, feasible_ratio, percent_optimal = maxclique_empirical_stats_from_samples(bitstrings, test_graphs[i], optimal_solution=test_solutions[i])

        if problem == 'mvc':
            exp_val, feasible_ratio, percent_optimal = mvc_empirical_stats_from_samples(bitstrings, test_graphs[i], optimal_solution=test_solutions[i])
            
        approximation_ratio = (exp_val) / test_solutions[i]
        approximation_ratio_list.append(approximation_ratio)
        all_feasible_ratios.append(feasible_ratio)
        percent_optimal_list.append(percent_optimal)

        print(f"Graph {i}: ExpVal = {exp_val:.4f}, OptVal = {test_solutions[i]:.4f}, FeasRate = {feasible_ratio:.4f}, PerOpt = {percent_optimal:.4f}")
        
    print(f" > Feasible Ratio: {(sum(all_feasible_ratios)/ len(all_feasible_ratios)):.4f}")
    print(f" > Percent Optimal: {(sum(percent_optimal_list)/ len(percent_optimal_list)):.4f}")

    print(f" > Approximation Ratio: {(sum(approximation_ratio_list)/ len(approximation_ratio_list)):.4f}")

    