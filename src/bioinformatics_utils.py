"""
Bioinformatics Utilities for Bayesian Network Analysis
Provides functions for data loading, network operations, and evaluation
"""

import pandas as pd
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_bioinformatics_data(file_path):
    """
    Load bioinformatics dataset with proper error handling
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Successfully loaded {file_path}")
        print(f"Dataset shape: {data.shape}")
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def create_bayesian_network(edges, node_names=None):
    """
    Create a Bayesian network graph from edge list
    """
    graph = nx.DiGraph()
    
    if node_names:
        graph.add_nodes_from(node_names)
    
    for edge in edges:
        if len(edge) >= 2:
            graph.add_edge(edge[0], edge[1])
            if len(edge) > 2:
                graph[edge[0]][edge[1]]['weight'] = edge[2]
    
    return graph

def calculate_conditional_probability(data, target, feature, target_val, feature_val):
    """
    Calculate conditional probability P(target=target_val | feature=feature_val)
    """
    try:
        # Count instances where both conditions are met
        numerator = len(data[(data[target] == target_val) & (data[feature] == feature_val)])
        
        # Count instances where feature condition is met
        denominator = len(data[data[feature] == feature_val])
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    except Exception as e:
        print(f"Error calculating conditional probability: {e}")
        return 0.0

def evaluate_network_performance(predictions, true_values):
    """
    Evaluate network performance using various metrics
    """
    accuracy = accuracy_score(true_values, predictions)
    conf_matrix = confusion_matrix(true_values, predictions)
    
    # Calculate additional metrics
    tn, fp, fn, tp = conf_matrix.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'confusion_matrix': conf_matrix
    }

def analyze_network_topology(graph):
    """
    Analyze network topology properties
    """
    if graph.number_of_nodes() == 0:
        return {}
    
    properties = {
        'nodes': graph.number_of_nodes(),
        'edges': graph.number_of_edges(),
        'density': nx.density(graph),
        'avg_degree': sum(dict(graph.degree()).values()) / graph.number_of_nodes(),
        'max_degree': max(dict(graph.degree()).values()),
        'avg_clustering': nx.average_clustering(graph),
        'connected_components': nx.number_connected_components(graph)
    }
    
    # Calculate diameter if graph is connected
    if nx.is_connected(graph):
        properties['diameter'] = nx.diameter(graph)
    else:
        properties['diameter'] = float('inf')
    
    return properties

def find_network_hubs(graph, top_k=5):
    """
    Find hub nodes in the network (nodes with highest degree)
    """
    degrees = dict(graph.degree())
    sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
    return sorted_nodes[:top_k]

def calculate_centrality_measures(graph):
    """
    Calculate various centrality measures for network nodes
    """
    centrality_measures = {}
    
    # Degree centrality
    centrality_measures['degree'] = nx.degree_centrality(graph)
    
    # Betweenness centrality
    centrality_measures['betweenness'] = nx.betweenness_centrality(graph)
    
    # Closeness centrality
    centrality_measures['closeness'] = nx.closeness_centrality(graph)
    
    # Eigenvector centrality
    try:
        centrality_measures['eigenvector'] = nx.eigenvector_centrality(graph, max_iter=1000)
    except:
        centrality_measures['eigenvector'] = {}
    
    return centrality_measures

def detect_network_communities(graph):
    """
    Detect communities in the network using various algorithms
    """
    communities = {}
    
    # Louvain method
    try:
        import community
        partition = community.best_partition(graph)
        communities['louvain'] = partition
    except ImportError:
        communities['louvain'] = {}
    
    # Label propagation
    try:
        communities['label_propagation'] = list(nx.community.label_propagation_communities(graph))
    except:
        communities['label_propagation'] = []
    
    # Girvan-Newman
    try:
        communities['girvan_newman'] = list(nx.community.girvan_newman(graph))
    except:
        communities['girvan_newman'] = []
    
    return communities

def calculate_network_robustness(graph, removal_fraction=0.1):
    """
    Calculate network robustness by removing nodes and measuring connectivity
    """
    n_nodes = graph.number_of_nodes()
    n_remove = int(n_nodes * removal_fraction)
    
    # Random removal
    nodes_to_remove = np.random.choice(list(graph.nodes()), n_remove, replace=False)
    graph_random = graph.copy()
    graph_random.remove_nodes_from(nodes_to_remove)
    
    # Targeted removal (highest degree nodes)
    degrees = dict(graph.degree())
    high_degree_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:n_remove]
    graph_targeted = graph.copy()
    graph_targeted.remove_nodes_from([node for node, _ in high_degree_nodes])
    
    robustness = {
        'random_removal_components': nx.number_connected_components(graph_random),
        'targeted_removal_components': nx.number_connected_components(graph_targeted),
        'original_components': nx.number_connected_components(graph)
    }
    
    return robustness

def prepare_data_for_bayesian_network(data, target_col, feature_cols=None):
    """
    Prepare data for Bayesian network analysis
    """
    if feature_cols is None:
        feature_cols = [col for col in data.columns if col != target_col]
    
    # Ensure all features are numeric
    numeric_data = data[feature_cols + [target_col]].copy()
    
    # Handle missing values
    numeric_data = numeric_data.fillna(numeric_data.mean())
    
    # Convert to binary if needed
    for col in feature_cols:
        if numeric_data[col].dtype in ['object', 'category']:
            numeric_data[col] = pd.Categorical(numeric_data[col]).codes
    
    return numeric_data

def split_data_for_validation(data, target_col, test_size=0.3, random_state=42):
    """
    Split data for training and validation
    """
    features = data.drop(columns=[target_col])
    target = data[target_col]
    
    return train_test_split(features, target, test_size=test_size, random_state=random_state)

def calculate_mutual_information(data, target_col, feature_cols=None):
    """
    Calculate mutual information between features and target
    """
    if feature_cols is None:
        feature_cols = [col for col in data.columns if col != target_col]
    
    mi_scores = {}
    
    for feature in feature_cols:
        try:
            # Calculate mutual information
            contingency_table = pd.crosstab(data[feature], data[target_col])
            total = contingency_table.sum().sum()
            
            mi = 0
            for i in range(len(contingency_table.index)):
                for j in range(len(contingency_table.columns)):
                    p_ij = contingency_table.iloc[i, j] / total
                    p_i = contingency_table.iloc[i, :].sum() / total
                    p_j = contingency_table.iloc[:, j].sum() / total
                    
                    if p_ij > 0 and p_i > 0 and p_j > 0:
                        mi += p_ij * np.log2(p_ij / (p_i * p_j))
            
            mi_scores[feature] = mi
        except:
            mi_scores[feature] = 0
    
    return mi_scores 