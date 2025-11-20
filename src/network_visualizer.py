"""
Network Visualization Utilities for Bayesian Network Analysis
Provides functions for visualizing networks and their properties
"""

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
import pandas as pd
import os
from datetime import datetime

def ensure_plots_directory():
    """
    Ensure the plots directory exists, create it if it doesn't
    """
    plots_dir = "plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        print(f"Created plots directory: {plots_dir}")
    return plots_dir

def generate_filename(prefix, title, extension="png"):
    """
    Generate a filename for saving plots
    """
    # Clean the title for filename
    clean_title = title.replace(" ", "_").replace(":", "").replace("(", "").replace(")", "")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{clean_title}_{timestamp}.{extension}"

def visualize_network(graph, title="Network Visualization", figsize=(12, 8), 
                     node_size=500, node_color='lightblue', edge_color='gray',
                     layout='spring', save_path=None):
    """
    Visualize a network graph
    """
    plt.figure(figsize=figsize)
    
    # Choose layout
    if layout == 'spring':
        pos = nx.spring_layout(graph, k=1, iterations=50)
    elif layout == 'circular':
        pos = nx.circular_layout(graph)
    elif layout == 'random':
        pos = nx.random_layout(graph)
    else:
        pos = nx.spring_layout(graph)
    
    # Draw the network
    nx.draw(graph, pos, 
            node_color=node_color,
            edge_color=edge_color,
            node_size=node_size,
            with_labels=True,
            font_size=8,
            font_weight='bold')
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Auto-save to plots directory if no save_path provided
    if save_path is None:
        plots_dir = ensure_plots_directory()
        filename = generate_filename("network", title)
        save_path = os.path.join(plots_dir, filename)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Network visualization saved to: {save_path}")
    
    plt.show()

def plot_network_metrics(graph, title="Network Metrics", figsize=(15, 10), save_path=None):
    """
    Plot various network metrics
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # 1. Degree distribution
    degrees = [d for n, d in graph.degree()]
    axes[0, 0].hist(degrees, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Degree Distribution')
    axes[0, 0].set_xlabel('Degree')
    axes[0, 0].set_ylabel('Frequency')
    
    # 2. Centrality measures
    centrality_measures = ['degree', 'betweenness', 'closeness']
    centrality_data = {}
    
    centrality_data['degree'] = list(nx.degree_centrality(graph).values())
    centrality_data['betweenness'] = list(nx.betweenness_centrality(graph).values())
    centrality_data['closeness'] = list(nx.closeness_centrality(graph).values())
    
    for i, measure in enumerate(centrality_measures):
        axes[0, 1].hist(centrality_data[measure], bins=15, alpha=0.6, label=measure.capitalize())
    axes[0, 1].set_title('Centrality Distributions')
    axes[0, 1].set_xlabel('Centrality Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    
    # 3. Clustering coefficient distribution
    clustering_coeffs = list(nx.clustering(graph).values())
    axes[0, 2].hist(clustering_coeffs, bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 2].set_title('Clustering Coefficient Distribution')
    axes[0, 2].set_xlabel('Clustering Coefficient')
    axes[0, 2].set_ylabel('Frequency')
    
    # 4. Network properties summary
    properties = {
        'Nodes': graph.number_of_nodes(),
        'Edges': graph.number_of_edges(),
        'Density': nx.density(graph),
        'Avg Degree': sum(dict(graph.degree()).values()) / graph.number_of_nodes(),
        'Avg Clustering': nx.average_clustering(graph),
        'Connected Components': nx.number_connected_components(graph)
    }
    
    y_pos = np.arange(len(properties))
    axes[1, 0].barh(y_pos, list(properties.values()), color='lightcoral')
    axes[1, 0].set_yticks(y_pos)
    axes[1, 0].set_yticklabels(properties.keys())
    axes[1, 0].set_title('Network Properties')
    axes[1, 0].set_xlabel('Value')
    
    # 5. Top nodes by degree
    degrees = dict(graph.degree())
    top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]
    node_names, node_degrees = zip(*top_nodes)
    
    axes[1, 1].barh(range(len(node_names)), node_degrees, color='gold')
    axes[1, 1].set_yticks(range(len(node_names)))
    axes[1, 1].set_yticklabels(node_names)
    axes[1, 1].set_title('Top 10 Nodes by Degree')
    axes[1, 1].set_xlabel('Degree')
    
    # 6. Edge weight distribution (if available)
    edge_weights = [data.get('weight', 1) for u, v, data in graph.edges(data=True)]
    if len(set(edge_weights)) > 1:  # Only plot if weights vary
        axes[1, 2].hist(edge_weights, bins=15, alpha=0.7, color='lightpink', edgecolor='black')
        axes[1, 2].set_title('Edge Weight Distribution')
        axes[1, 2].set_xlabel('Weight')
        axes[1, 2].set_ylabel('Frequency')
    else:
        axes[1, 2].text(0.5, 0.5, 'No weight variation', ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Edge Weights')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Auto-save to plots directory if no save_path provided
    if save_path is None:
        plots_dir = ensure_plots_directory()
        filename = generate_filename("metrics", title)
        save_path = os.path.join(plots_dir, filename)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Network metrics plot saved to: {save_path}")
    
    plt.show()

def plot_community_analysis(graph, communities, title="Community Analysis", figsize=(15, 10), save_path=None):
    """
    Plot community analysis results
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Community size distribution
    community_sizes = [len(community) for community in communities]
    axes[0, 0].hist(community_sizes, bins=min(10, len(community_sizes)), alpha=0.7, color='lightblue', edgecolor='black')
    axes[0, 0].set_title('Community Size Distribution')
    axes[0, 0].set_xlabel('Community Size')
    axes[0, 0].set_ylabel('Frequency')
    
    # 2. Number of communities vs modularity
    try:
        modularity = nx.community.modularity(graph, communities)
        axes[0, 1].bar(['Modularity'], [modularity], color='lightgreen')
        axes[0, 1].set_title(f'Network Modularity: {modularity:.3f}')
        axes[0, 1].set_ylabel('Modularity')
    except:
        axes[0, 1].text(0.5, 0.5, 'Modularity calculation failed', ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Modularity')
    
    # 3. Community visualization
    pos = nx.spring_layout(graph)
    colors = plt.cm.Set3(np.linspace(0, 1, len(communities)))
    
    for i, community in enumerate(communities):
        nx.draw_networkx_nodes(graph, pos, nodelist=list(community), 
                              node_color=[colors[i]], node_size=300, ax=axes[1, 0])
    
    nx.draw_networkx_edges(graph, pos, alpha=0.3, ax=axes[1, 0])
    axes[1, 0].set_title('Community Structure')
    axes[1, 0].axis('off')
    
    # 4. Community statistics
    if len(communities) > 0:
        avg_community_size = np.mean([len(c) for c in communities])
        max_community_size = max([len(c) for c in communities])
        min_community_size = min([len(c) for c in communities])
        
        stats = {
            'Number of Communities': len(communities),
            'Avg Community Size': avg_community_size,
            'Max Community Size': max_community_size,
            'Min Community Size': min_community_size
        }
        
        y_pos = np.arange(len(stats))
        axes[1, 1].barh(y_pos, list(stats.values()), color='lightcoral')
        axes[1, 1].set_yticks(y_pos)
        axes[1, 1].set_yticklabels(stats.keys())
        axes[1, 1].set_title('Community Statistics')
        axes[1, 1].set_xlabel('Value')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Auto-save to plots directory if no save_path provided
    if save_path is None:
        plots_dir = ensure_plots_directory()
        filename = generate_filename("community", title)
        save_path = os.path.join(plots_dir, filename)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Community analysis plot saved to: {save_path}")
    
    plt.show()

def plot_correlation_heatmap(data, title="Feature Correlation Heatmap", figsize=(12, 10), save_path=None):
    """
    Plot correlation heatmap for features
    """
    plt.figure(figsize=figsize)
    
    # Calculate correlation matrix
    corr_matrix = data.corr()
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Auto-save to plots directory if no save_path provided
    if save_path is None:
        plots_dir = ensure_plots_directory()
        filename = generate_filename("correlation", title)
        save_path = os.path.join(plots_dir, filename)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Correlation heatmap saved to: {save_path}")
    
    plt.show()

def plot_feature_importance(importance_scores, title="Feature Importance", figsize=(12, 8), save_path=None):
    """
    Plot feature importance scores
    """
    plt.figure(figsize=figsize)
    
    # Sort features by importance
    sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
    features, scores = zip(*sorted_features)
    
    # Create bar plot
    colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
    bars = plt.barh(range(len(features)), scores, color=colors)
    
    plt.yticks(range(len(features)), features)
    plt.xlabel('Importance Score')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.gca().invert_yaxis()
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, scores)):
        plt.text(score + 0.01, i, f'{score:.3f}', va='center')
    
    plt.tight_layout()
    
    # Auto-save to plots directory if no save_path provided
    if save_path is None:
        plots_dir = ensure_plots_directory()
        filename = generate_filename("importance", title)
        save_path = os.path.join(plots_dir, filename)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to: {save_path}")
    
    plt.show()

def plot_performance_metrics(metrics, title="Model Performance Metrics", figsize=(12, 8), save_path=None):
    """
    Plot model performance metrics
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Accuracy
    if 'accuracy' in metrics:
        axes[0, 0].bar(['Accuracy'], [metrics['accuracy']], color='lightgreen')
        axes[0, 0].set_title(f'Accuracy: {metrics["accuracy"]:.3f}')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_ylim(0, 1)
    
    # 2. Confusion Matrix
    if 'confusion_matrix' in metrics:
        cm = metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
        axes[0, 1].set_title('Confusion Matrix')
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Actual')
    
    # 3. Additional metrics
    additional_metrics = ['sensitivity', 'specificity', 'precision']
    metric_values = []
    metric_names = []
    
    for metric in additional_metrics:
        if metric in metrics:
            metric_values.append(metrics[metric])
            metric_names.append(metric.capitalize())
    
    if metric_values:
        axes[1, 0].bar(metric_names, metric_values, color=['lightblue', 'lightcoral', 'lightyellow'])
        axes[1, 0].set_title('Additional Metrics')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_ylim(0, 1)
        plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)
    
    # 4. ROC curve placeholder (if available)
    axes[1, 1].text(0.5, 0.5, 'ROC Curve\n(if available)', ha='center', va='center', transform=axes[1, 1].transAxes)
    axes[1, 1].set_title('ROC Curve')
    axes[1, 1].set_xlabel('False Positive Rate')
    axes[1, 1].set_ylabel('True Positive Rate')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Auto-save to plots directory if no save_path provided
    if save_path is None:
        plots_dir = ensure_plots_directory()
        filename = generate_filename("performance", title)
        save_path = os.path.join(plots_dir, filename)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Performance metrics plot saved to: {save_path}")
    
    plt.show()

def create_network_animation(graph, frames=10, title="Network Evolution", save_path=None):
    """
    Create a simple animation of network evolution (placeholder)
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for frame in range(frames):
        ax.clear()
        
        # Create subgraph with increasing number of nodes
        nodes_to_include = list(graph.nodes())[:int(len(graph.nodes()) * (frame + 1) / frames)]
        subgraph = graph.subgraph(nodes_to_include)
        
        pos = nx.spring_layout(subgraph)
        nx.draw(subgraph, pos, node_color='lightblue', node_size=300, 
                with_labels=True, font_size=8, ax=ax)
        
        ax.set_title(f'{title} - Frame {frame + 1}/{frames}')
        plt.pause(0.5)
    
    # Auto-save to plots directory if no save_path provided
    if save_path is None:
        plots_dir = ensure_plots_directory()
        filename = generate_filename("animation", title)
        save_path = os.path.join(plots_dir, filename)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Network animation saved to: {save_path}")
    
    plt.show() 