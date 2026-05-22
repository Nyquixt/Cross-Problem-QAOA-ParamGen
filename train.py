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
import time
from copy import deepcopy

from rnn import ConditionedLSTM, LSTMNet
from dataset import collate_fn, generate_graphs, GraphDataset
from qaoa import QAOAMaxcutCost, QAOAMISCost, QAOAMaxCliqueCost, QAOAMVCCost
from utils import train_step_batch_parallel, train_step_g2v_batch_parallel, train_step_batch, train_step_g2v_batch

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset", type=str, default="toy")
    parser.add_argument("--problem", type=str, default="maxcut")
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
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

    # toy dataset
    train_path = f'./dataset_{dataset}/trainset/graphs.pkl'
    if use_g2v:
        train_embedding_path = f'./dataset_{dataset}/trainset/embeddings.npy'
    else:
        train_embedding_path = f'./dataset_{dataset}/trainset/{problem}_unihetco_embeddings.pkl'

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
    if os.path.exists(train_path):
        print("Found train graphs pkl, Loading...")
        with open(train_path, 'rb') as infile:
            train_graphs = pickle.load(infile)
    else:
        train_graphs = generate_graphs(n_graphs=500, n_nodes_range=[6, 9], seed=42)
        with open(train_path, 'wb') as outfile:
            pickle.dump(train_graphs, outfile)

    if problem == 'maxcut':
        train_graph_cost_list = [QAOAMaxcutCost(g, n_layers=n_layers, normalized=normalize) for g in train_graphs]
    elif problem == 'mis':
        train_graph_cost_list = [QAOAMISCost(g, n_layers=n_layers, normalized=normalize) for g in train_graphs]
    elif problem == 'maxclique':
        train_graph_cost_list = [QAOAMaxCliqueCost(g, n_layers=n_layers, normalized=normalize) for g in train_graphs]
    elif problem == 'mvc':
        train_graph_cost_list = [QAOAMVCCost(g, n_layers=n_layers, normalized=normalize) for g in train_graphs]

    if use_g2v or use_uni:
        if use_uni:
            with open(train_embedding_path, "rb") as infile:
                train_embeddings = pickle.load(infile)
        else:
            train_embeddings = np.load(train_embedding_path)
        
        lstm = ConditionedLSTM(p=n_layers, device=device, graph_dim=train_embeddings.shape[-1], T=horizon).to(device)
        
        train_embeddings = [row.copy() for row in train_embeddings]  # each is a (D,) np array
    else:
        lstm = LSTMNet(p=n_layers, device=device, T=horizon).to(device)
        
    optimizer = optim.Adam(lstm.parameters(), lr=0.001)

    train_dataset = GraphDataset(train_graphs, train_graph_cost_list, train_embeddings if (use_g2v or use_uni) else None)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=8)

    best_loss = np.inf
    best_model = None
    
    for epoch in range(epochs):
        start_time = time.perf_counter()
        print(f"Epoch {epoch+1}")
        total_loss_list = []
        for i, batch in enumerate(train_dataloader):
            graph_costs = batch["graph_costs"]
            
            # Perform a single training step
            if use_g2v or use_uni:
                embedding = batch["embeddings"]
                if batch_size > 1:
                    loss = train_step_g2v_batch_parallel(
                        graph_costs, embedding, lstm, optimizer,
                        n_jobs=max(1, os.cpu_count())
                    )  # returns a PyTorch scalar tensor
                else:
                    loss = train_step_g2v_batch(
                        graph_costs, embedding, lstm, optimizer
                    )  # returns a PyTorch scalar tensor
            else:
                if batch_size > 1:
                    loss = train_step_batch_parallel(
                        graph_costs, lstm, optimizer,
                        n_jobs=max(1, os.cpu_count())                    
                    )  # returns a PyTorch scalar tensor
                else:
                    loss = train_step_batch(
                        graph_costs, lstm, optimizer
                    )  # returns a PyTorch scalar tensor
            # Convert to a Python float for logging
            loss_value = loss.item()
            total_loss_list.append(loss_value)

            if loss_value < best_loss:
                best_loss = loss_value
                best_model = deepcopy(lstm) # remember to change torch.save line
                torch.save(best_model.state_dict(), f"./models_{args.dataset}/{problem}_p={n_layers}_T={horizon}_epochs={epochs}_bs={batch_size}_g2v={use_g2v}_uni={use_uni}_normalize={normalize}_seed={seed}.pth")

            # Log every n batches (or as desired)
            if i % 10 == 0:
                print(f" > Batch {i+1}/{len(train_graph_cost_list) // batch_size} - Loss: {loss_value:.4f}")

        # Compute average (mean) loss across all graphs in this epoch
        mean_loss = np.mean(total_loss_list)
        print(f" >> Mean Loss during epoch: {mean_loss:.4f}")
        end_time = time.perf_counter()

        # Calculate the duration
        elapsed_time = end_time - start_time
        print(f"Execution took: {elapsed_time:.4f} seconds")

    torch.save(best_model.state_dict(), f"./models_{args.dataset}/{problem}_p={n_layers}_T={horizon}_epochs={epochs}_bs={batch_size}_g2v={use_g2v}_uni={use_uni}_normalize={normalize}_seed={seed}.pth")