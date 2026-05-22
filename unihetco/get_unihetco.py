import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
from dataset import FixedTrainDataset, FixedTestDataset, collate_fn
from model import UnifiedQPModel
from utils import finetune_one_graph

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    # problems, datasets args
    parser.add_argument("--dataset", type=str, default="twitter")
    parser.add_argument("--problem", type=str, required=True)
    # model args
    parser.add_argument("--qc_layer_name", type=str, default="GraphConv")
    parser.add_argument("--ab_layer_name", type=str, default="GraphConv")
    parser.add_argument("--num_graph_layers", type=int, default=6)
    parser.add_argument("--num_qc_layers", type=int, default=2)
    parser.add_argument("--num_ab_layers", type=int, default=1)
    # train args
    parser.add_argument("--normalize", action='store_true')
    parser.add_argument("--attn", action='store_true')
    parser.add_argument("--graph_embed", action='store_true')
    parser.add_argument("--lambda_k", type=float, default=0.5)
    parser.add_argument("--lambda_f", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=16)
    parser.add_argument("--n_updates", type=int, default=5)
    parser.add_argument("--val_every_n_epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8)
    # misc args
    parser.add_argument("--exp_name", type=str, default='erm_multi')
    parser.add_argument("--result_folder", type=str, default='./results_twitter')
    parser.add_argument("--log", action='store_true')
    parser.add_argument("--ft", action='store_true')

    args = parser.parse_args()

    # set seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # problems, datasets args
    dataset = args.dataset
    problem = args.problem

    qc_layer_name = args.qc_layer_name
    ab_layer_name = args.ab_layer_name
    num_graph_layers = args.num_graph_layers
    num_qc_layers = args.num_qc_layers
    num_ab_layers = args.num_ab_layers

    lambda_k = args.lambda_k
    lambda_f = args.lambda_f
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    n_updates = args.n_updates
    val_every_n_epochs = args.val_every_n_epochs
    num_workers = args.num_workers

    exp_name = args.exp_name
    result_folder = os.path.join(args.result_folder, exp_name)

    # test dataset
    with open(f"../dataset_l2l/testset/graphs.pkl", "rb") as infile:
        test_graphs = pickle.load(infile)
    test_dataset = FixedTrainDataset(test_graphs, problem=problem, base_seed=seed) # v3

    with open(f"../dataset_l2l/trainset/graphs.pkl", "rb") as infile:
        train_graphs = pickle.load(infile)        
    train_dataset = FixedTrainDataset(train_graphs, problem=problem, base_seed=seed)
        
    # loader
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, persistent_workers=True, collate_fn=collate_fn, num_workers=num_workers)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=False, persistent_workers=True, collate_fn=collate_fn, num_workers=num_workers)

    # load state_dict
    print("Loading model...")
    state_dict_folder = os.path.join("./models", exp_name)
    state_dict_file = os.path.join(state_dict_folder, "model.pth")
    state_dict = torch.load(state_dict_file, weights_only=True, map_location=torch.device(device))
    
    best_model = UnifiedQPModel(
        graph_in_channels=4,
        qc_in_channels=1,
        ab_in_channels=(1, 1), # constr, var  
        hidden_channels=hidden_size, 
        out_channels=1,
        use_attn=args.attn, 
        use_graph_embed=args.graph_embed,
        n_graph_layers=num_graph_layers, 
        n_qc_layers=num_qc_layers, 
        n_ab_layers=num_ab_layers,
        qc_layer_name=qc_layer_name,
        ab_layer_name=ab_layer_name,
        return_features=True
    ).to(device)


    best_model.load_state_dict(state_dict)
    
    # testing
    train_graph_features_list = []
    
    best_model.eval()
    for idx, batch in enumerate(train_loader): # test        
        problem_graph_batch = batch["problem_graph_batch"].to(device)
        qc_graph_batch = batch["qc_graph_batch"].to(device)
        if batch["ab_graph_batch"] is None:
            ab_graph_batch = None
        else:
            ab_graph_batch = batch["ab_graph_batch"].to(device)
        
        node_features = best_model(problem_graph_batch, qc_graph_batch, ab_graph_batch)
        graph_features = node_features.mean(dim=0)
        train_graph_features_list.append(graph_features.detach().cpu())

    train_graph_features = torch.stack(train_graph_features_list).numpy()
    print(train_graph_features.shape)


    test_graph_features_list = []
    
    best_model.eval()
    for idx, batch in enumerate(test_loader): # test        
        problem_graph_batch = batch["problem_graph_batch"].to(device)
        qc_graph_batch = batch["qc_graph_batch"].to(device)
        if batch["ab_graph_batch"] is None:
            ab_graph_batch = None
        else:
            ab_graph_batch = batch["ab_graph_batch"].to(device)
        
        node_features = best_model(problem_graph_batch, qc_graph_batch, ab_graph_batch)
        graph_features = node_features.mean(dim=0)
        test_graph_features_list.append(graph_features.detach().cpu())

    test_graph_features = torch.stack(test_graph_features_list).numpy()
    print(test_graph_features.shape)

    with open(f'../dataset_l2l/trainset/{problem}_unihetco_embeddings.pkl', "wb") as outfile:
        pickle.dump(train_graph_features, outfile)
    
    with open(f'../dataset_l2l/testset/{problem}_unihetco_embeddings.pkl', "wb") as outfile:
        pickle.dump(test_graph_features, outfile)