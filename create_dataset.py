import os
from dataset import generate_graphs
import pickle

dataset_name = 'l2l'
data_folder = f'dataset_{dataset_name}'
train_folder = os.path.join(data_folder, 'trainset')
train_path = os.path.join(train_folder, 'graphs.pkl')

test_folder = os.path.join(data_folder, 'testset')
test_path = os.path.join(test_folder, 'graphs.pkl')

if not os.path.exists(data_folder):
    os.mkdir(data_folder)
if not os.path.exists(train_folder):
    os.makedirs(train_folder)
if not os.path.exists(test_folder):
    os.makedirs(test_folder)

if os.path.exists(train_path):
    print("Found train graphs pkl, Loading...")
    with open(train_path, 'rb') as infile:
        train_graphs = pickle.load(infile)
else:
    train_graphs = generate_graphs(n_graphs=1000, n_nodes_range=[6, 10], seed=42)
    with open(train_path, 'wb') as outfile:
        pickle.dump(train_graphs, outfile)

if os.path.exists(test_path):
    print("Found test graphs pkl, Loading...")
    with open(test_path, 'rb') as infile:
        test_graphs = pickle.load(infile)
else:
    test_graphs = generate_graphs(n_graphs=100, n_nodes_range=[12, 12], seed=100)
    with open(test_path, 'wb') as outfile:
        pickle.dump(test_graphs, outfile)


