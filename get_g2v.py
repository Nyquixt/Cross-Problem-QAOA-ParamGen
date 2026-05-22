import os
import pickle
from karateclub import Graph2Vec 
from dataset import generate_graphs
import numpy as np

dataset_name = 'l2l'
data_folder = f'dataset_{dataset_name}'
train_folder = os.path.join(data_folder, 'trainset')
train_path = os.path.join(train_folder, 'graphs.pkl')
train_embedding_path = os.path.join(train_folder, 'embeddings.npy')

test_folder = os.path.join(data_folder, 'testset')
test_path = os.path.join(test_folder, 'graphs.pkl')
test_embedding_path = os.path.join(test_folder, 'embeddings.npy')

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
    test_graphs = generate_graphs(n_graphs=100, n_nodes_range=[12, 12], seed=50)
    with open(test_path, 'wb') as outfile:
        pickle.dump(test_graphs, outfile)

print("Fitting Graph2Vec...")
g2v = Graph2Vec(dimensions=48)
g2v.fit(train_graphs)
train_embeddings = g2v.infer(train_graphs) # N_graphs x 48
test_embeddings = g2v.infer(test_graphs) # N_graphs x 48

print(type(train_embeddings), type(test_embeddings))

np.save(train_embedding_path, train_embeddings)
np.save(test_embedding_path, test_embeddings)