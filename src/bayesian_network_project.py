#!/usr/bin/env python3
"""
Bayesian Network and Reasoning in Bioinformatics
Programming Assignment: CSCI 384 AI - Advanced Machine Learning

STUDENT VERSION - Complete the TODO sections below!

This project implements Bayesian Networks for bioinformatics applications including:
- Gene expression analysis and disease prediction
- Genetic marker analysis for diabetes risk assessment
- Protein-protein interaction network analysis

Difficulty Level: 6/10
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import visualization functions (students can use these)
from src.network_visualizer import visualize_network, plot_network_metrics

# Global variables for grading script access
gene_conditional_probs = {}
disease_conditional_probs = {}

print("Bayesian Network and Reasoning in Bioinformatics")
print("=" * 60)

# ============================================================================
# [10 pts] STEP 1: Load and Explore Bioinformatics Datasets
# ============================================================================

print("\nSTEP 1: Loading and Exploring Bioinformatics Datasets")
print("-" * 50)

# TODO: Load the three datasets
# HINT: Use pd.read_csv() to load the CSV files from the data/ folder
gene_data = None  # TODO: Load gene_expression.csv
disease_data = None  # TODO: Load disease_markers.csv (now contains real SNP names)
protein_data = None  # TODO: Load protein_interactions.csv

# TODO: Print basic information about each dataset
# HINT: Use .shape, .columns, and .value_counts() to explore the data
print(f"Gene Expression Dataset Shape: {gene_data.shape if gene_data is not None else 'Not loaded'}")
# TODO: Add more exploration code here

# TODO: Calculate basic statistics for gene expression data
# HINT: Use .describe() to get statistical summary
gene_stats = None  # TODO: Calculate statistics
print(f"\nGene Expression Statistics:")
# TODO: Add statistics printing code

# TODO: Calculate correlation with disease status
# HINT: Use .corrwith() to find correlations between genes and disease status
# HINT: Use .abs().nlargest(5) to find top 5 correlated genes
gene_correlations = None  # TODO: Calculate correlations
top_correlated_genes = None  # TODO: Find top 5 correlated genes
print(f"\nTop 5 genes correlated with disease status:")
# TODO: Add correlation printing code

# ============================================================================
# [15 pts] STEP 2: Data Preprocessing and Feature Engineering
# ============================================================================

print("\nSTEP 2: Data Preprocessing and Feature Engineering")
print("-" * 50)

# TODO: Prepare gene expression data for analysis
# HINT: Separate features from target variable
gene_features = None  # TODO: Extract features
gene_target = None  # TODO: Extract target

# TODO: Normalize gene expression data
# HINT: Use StandardScaler() to normalize the features
# HINT: Use fit_transform() and create a DataFrame with original column names
scaler = None  # TODO: Create scaler
gene_features_scaled = None  # TODO: Scale features

# TODO: Create binary features for high/low expression
# HINT: Use (gene_features_scaled > 0).astype(int) to create binary features
# HINT: Rename columns to add "_high" suffix
gene_features_binary = None  # TODO: Create binary features

# TODO: Combine original and binary features
# HINT: Use pd.concat() to combine scaled and binary features
gene_features_combined = None  # TODO: Combine features
print(f"Combined feature set shape: {gene_features_combined.shape if gene_features_combined is not None else 'Not implemented'}")

# TODO: Prepare disease markers data
# HINT: Separate features from target variable
disease_features = None  # TODO: Extract disease features
disease_target = None  # TODO: Extract disease target

# TODO: Create interaction features for SNPs
# HINT: Find SNP columns using list comprehension with .startswith('rs')
# NOTE: The dataset now contains real SNP names like rs7903146, rs12255372, etc.
snp_columns = None  # TODO: Find SNP columns
clinical_columns = ['age', 'bmi', 'glucose', 'insulin', 'hdl_cholesterol']

# TODO: Create SNP interaction features
# HINT: Use nested loops to create pairwise interactions
# HINT: Multiply SNP values: disease_features[snp_columns[i]] * disease_features[snp_columns[j]]
snp_interactions = None  # TODO: Create SNP interactions

# TODO: Combine clinical and SNP features
# HINT: Use pd.concat() to combine original features with interaction features
disease_features_combined = None  # TODO: Combine disease features
print(f"Disease features combined shape: {disease_features_combined.shape if disease_features_combined is not None else 'Not implemented'}")

# ============================================================================
# [18 pts] STEP 3: Bayesian Network Structure Learning
# ============================================================================

print("\nSTEP 3: Bayesian Network Structure Learning")
print("-" * 50)

# TODO: Implement structure learning algorithm
# HINT: Use correlation-based approach with a threshold
# HINT: Calculate absolute correlations and find edges above threshold
def learn_bayesian_structure(data, threshold=0.3):
    """
    Learn Bayesian network structure using correlation-based approach
    
    Args:
        data: DataFrame with features
        threshold: Correlation threshold for creating edges
    
    Returns:
        list: List of tuples (node1, node2, correlation)
    """
    # TODO: Calculate correlation matrix
    # HINT: Use data.corr().abs() to get absolute correlations
    correlations = None  # TODO: Calculate correlations
    edges = []
    
    # TODO: Find edges above threshold
    # HINT: Use nested loops to check each pair of features
    # HINT: Only add edges if correlation > threshold
    # TODO: Add your loop code here
    
    return edges

# TODO: Learn structure for gene expression network
# HINT: Select a subset of genes for network analysis
# HINT: Use genes from different modules: ['CDK1', 'MAPK1', 'PIK3CA', 'BCL2', 'GLUT1', 'MYC']
selected_genes = ['CDK1', 'MAPK1', 'PIK3CA', 'BCL2', 'GLUT1', 'MYC']
gene_subset = None  # TODO: Select gene subset
gene_network_edges = None  # TODO: Learn gene network structure
print(f"Gene network edges found: {len(gene_network_edges) if gene_network_edges is not None else 'Not implemented'}")

# TODO: Learn structure for disease markers network
# HINT: Use first 15 columns of disease_features_combined
# HINT: Use a lower threshold (0.1) for disease markers
disease_network_edges = None  # TODO: Learn disease network structure
print(f"Disease network edges found: {len(disease_network_edges) if disease_network_edges is not None else 'Not implemented'}")

# TODO: Create network graphs
# HINT: Use nx.Graph() to create NetworkX graphs
# HINT: Add edges with weights from correlation values
gene_graph = None  # TODO: Create gene graph
disease_graph = None  # TODO: Create disease graph

print(f"Gene network nodes: {gene_graph.number_of_nodes() if gene_graph is not None else 'Not created'}, edges: {gene_graph.number_of_edges() if gene_graph is not None else 'Not created'}")
print(f"Disease network nodes: {disease_graph.number_of_nodes() if disease_graph is not None else 'Not created'}, edges: {disease_graph.number_of_edges() if disease_graph is not None else 'Not created'}")

# ============================================================================
# [15 pts] STEP 4: Conditional Probability Calculations
# ============================================================================

print("\nSTEP 4: Conditional Probability Calculations")
print("-" * 50)

# TODO: Implement conditional probability calculation function
# HINT: Calculate P(target|feature) for binary features
# HINT: Use conditional probability formula: P(A|B) = P(A∩B) / P(B)
def calculate_conditional_probabilities(data, target_col, feature_cols):
    """
    Calculate conditional probabilities P(target|feature) for binary features
    
    Args:
        data: DataFrame with features and target
        target_col: Name of target column
        feature_cols: List of feature column names
    
    Returns:
        dict: Dictionary of conditional probabilities
    """
    conditional_probs = {}
    
    # TODO: Loop through each feature
    for feature in feature_cols:
        if data[feature].dtype in ['int64', 'bool']:
            # TODO: Calculate conditional probabilities for each feature value and target value
            # HINT: Use len() and boolean indexing to count occurrences
            # HINT: Calculate P(target=val1|feature=val2) for all combinations
            for feature_val in [0, 1]:
                for target_val in [0, 1]:
                    # TODO: Calculate conditional probability
                    # HINT: Count samples where both conditions are met
                    # HINT: Divide by count of samples where feature condition is met
                    prob = 0.0  # TODO: Calculate probability
                    conditional_probs[f"P({target_col}={target_val}|{feature}={feature_val})"] = prob
    
    return conditional_probs

# TODO: Calculate conditional probabilities for gene expression
# HINT: Use first 5 binary features
gene_binary_features = None  # TODO: Select binary features
gene_conditional_probs = None  # TODO: Calculate conditional probabilities

print("Conditional probabilities for gene expression:")
# TODO: Add probability printing code

# TODO: Calculate conditional probabilities for disease markers
# HINT: Ensure SNP columns are binary (0/1) before calculation
# HINT: Use first 5 SNP columns from snp_columns
disease_binary_features = None  # TODO: Select disease binary features

# TODO: Binarize SNP columns if needed
# HINT: Check if values are binary, if not convert to binary
# HINT: Use (disease_binary_features[col] > 0).astype(int) to binarize
print("\nStep 4c: Checking SNP columns for binary values:")
# TODO: Add SNP checking code

disease_data_combined = None  # TODO: Combine disease data
disease_conditional_probs = None  # TODO: Calculate disease conditional probabilities
# NOTE: disease_conditional_probs should be a dictionary for grading script access

print(f"Disease binary features shape: {disease_binary_features.shape if disease_binary_features is not None else 'Not implemented'}")
print(f"Disease target shape: {disease_target.shape if disease_target is not None else 'Not implemented'}")
print(f"SNP columns: {snp_columns[:5] if snp_columns is not None else 'Not implemented'}")

print("\nConditional probabilities for disease markers:")
# TODO: Add disease probability printing code

# ============================================================================
# [18 pts] STEP 5: Probabilistic Inference
# ============================================================================

print("\nSTEP 5: Probabilistic Inference")
print("-" * 50)

# TODO: Implement Naive Bayes inference function
# HINT: Use the conditional probabilities and prior probabilities
# HINT: Calculate likelihood for each class and apply Bayes' theorem
def naive_bayes_inference(features, conditional_probs, prior_probs):
    """
    Perform naive Bayes inference
    
    Args:
        features: DataFrame of features
        conditional_probs: Dictionary of conditional probabilities
        prior_probs: List of prior probabilities [P(class=0), P(class=1)]
    
    Returns:
        list: List of predicted classes
    """
    predictions = []
    
    # TODO: Loop through each sample
    for _, row in features.iterrows():
        # TODO: Calculate likelihood for each class
        # HINT: Start with likelihood_0 = 1.0 and likelihood_1 = 1.0
        likelihood_0 = 1.0  # TODO: Initialize
        likelihood_1 = 1.0  # TODO: Initialize
        
        # TODO: Loop through each feature
        for feature in features.columns:
            feature_val = row[feature]
            
            # TODO: Get conditional probabilities from dictionary
            # HINT: Use .get() method with default value 0.5
            prob_0 = 0.5  # TODO: Get probability for class 0
            prob_1 = 0.5  # TODO: Get probability for class 1
            
            # TODO: Multiply likelihoods
            likelihood_0 *= prob_0  # TODO: Update likelihood_0
            likelihood_1 *= prob_1  # TODO: Update likelihood_1
        
        # TODO: Apply prior probabilities
        # HINT: Multiply likelihood by prior: posterior = likelihood * prior
        posterior_0 = 0.0  # TODO: Calculate posterior for class 0
        posterior_1 = 0.0  # TODO: Calculate posterior for class 1
        
        # TODO: Normalize probabilities
        # HINT: Divide by sum of posteriors
        total = 0.0  # TODO: Calculate total
        posterior_0 /= total  # TODO: Normalize
        posterior_1 /= total  # TODO: Normalize
        
        # TODO: Make prediction
        # HINT: Choose class with higher posterior probability
        predictions.append(0)  # TODO: Make prediction
    
    return predictions

# TODO: Split data for inference
# HINT: Use train_test_split() with test_size=0.3 and random_state=42
X_train = None  # TODO: Split training features
X_test = None  # TODO: Split test features
y_train = None  # TODO: Split training targets
y_test = None  # TODO: Split test targets

# TODO: Calculate prior probabilities
# HINT: Count samples in each class and divide by total
prior_probs = [0.0, 0.0]  # TODO: Calculate prior probabilities

# TODO: Perform inference
# HINT: Call naive_bayes_inference() with test features, conditional probabilities, and priors
predictions = None  # TODO: Perform inference

# TODO: Calculate accuracy
# HINT: Use accuracy_score() from sklearn.metrics
accuracy = 0.0  # TODO: Calculate accuracy
print(f"Naive Bayes inference accuracy: {accuracy:.3f}")

# ============================================================================
# [10 pts] STEP 6: Network Analysis and Visualization
# ============================================================================

print("\nSTEP 6: Network Analysis and Visualization")
print("-" * 50)

# TODO: Implement network analysis function
# HINT: Calculate network properties like density, degree, clustering coefficient
def analyze_network_properties(graph):
    """
    Analyze network properties
    
    Args:
        graph: NetworkX graph object
    
    Returns:
        dict: Dictionary of network properties
    """
    properties = {}
    
    # TODO: Calculate basic properties
    # HINT: Use graph.number_of_nodes(), graph.number_of_edges(), nx.density()
    properties['nodes'] = 0  # TODO: Count nodes
    properties['edges'] = 0  # TODO: Count edges
    properties['density'] = 0.0  # TODO: Calculate density
    
    # TODO: Calculate centrality measures
    # HINT: Use dict(graph.degree()) to get degree dictionary
    if graph.number_of_nodes() > 0:
        # TODO: Calculate average degree
        # HINT: Sum all degrees and divide by number of nodes
        properties['avg_degree'] = 0.0  # TODO: Calculate average degree
        
        # TODO: Calculate maximum degree
        # HINT: Use max() on degree values
        properties['max_degree'] = 0  # TODO: Calculate max degree
        
        # TODO: Calculate clustering coefficient
        # HINT: Use nx.average_clustering(graph)
        properties['avg_clustering'] = 0.0  # TODO: Calculate clustering
        
        # TODO: Calculate number of connected components
        # HINT: Use nx.number_connected_components(graph)
        properties['connected_components'] = 0  # TODO: Count components
    
    return properties

# TODO: Analyze gene network
# HINT: Call analyze_network_properties() with gene_graph
gene_properties = None  # TODO: Analyze gene network
print("Gene Network Properties:")
# TODO: Add property printing code

# TODO: Analyze disease network
# HINT: Call analyze_network_properties() with disease_graph
disease_properties = None  # TODO: Analyze disease network
print("\nDisease Network Properties:")
# TODO: Add property printing code

# TODO: Visualize networks
# HINT: Use visualize_network() and plot_network_metrics() functions
# HINT: Call visualize_network(gene_graph, title="Gene Expression Network")
# HINT: Call plot_network_metrics(gene_graph, title="Gene Network Metrics")
print("\nVisualizing networks...")
# TODO: Add visualization code for gene network
# TODO: Add visualization code for disease network

# ============================================================================
# [10 pts] STEP 7: Protein Interaction Network Analysis
# ============================================================================

print("\nSTEP 7: Protein Interaction Network Analysis")
print("-" * 50)

# TODO: Create protein interaction network
# HINT: Use nx.Graph() to create a new graph
protein_graph = None  # TODO: Create protein graph

# TODO: Add edges from protein interaction data
# HINT: Loop through protein_data rows
# HINT: Add edges with protein names and interaction scores as weights
# TODO: Add your loop code here

# TODO: Analyze protein network
# HINT: Use the same analyze_network_properties() function
protein_properties = None  # TODO: Analyze protein network
print("Protein Interaction Network Properties:")
# TODO: Add property printing code

# TODO: Find hub proteins (high degree nodes)
# HINT: Use dict(protein_graph.degree()) to get degree dictionary
# HINT: Sort by degree in descending order and take top 5
protein_degrees = None  # TODO: Get protein degrees
hub_proteins = None  # TODO: Find hub proteins
print(f"\nTop 5 hub proteins:")
# TODO: Add hub protein printing code

# TODO: Analyze interaction types
# HINT: Use .value_counts() on interaction_type column
interaction_types = None  # TODO: Analyze interaction types
print(f"\nInteraction type distribution:")
# TODO: Add interaction type printing code

# TODO: Visualize protein network
# HINT: Use visualize_network() and plot_network_metrics() functions
# HINT: Call visualize_network(protein_graph, title="Protein Interaction Network")
# HINT: Call plot_network_metrics(protein_graph, title="Protein Network Metrics")
print("\nVisualizing protein interaction network...")
# TODO: Add visualization code for protein network

# ============================================================================
# [4 pts] STEP 8: Model Evaluation and Biological Interpretation
# ============================================================================

print("\nSTEP 8: Model Evaluation and Biological Interpretation")
print("-" * 50)

# TODO: Evaluate gene expression model
# HINT: Use confusion_matrix() and classification_report() from sklearn.metrics
confusion_mat = None  # TODO: Calculate confusion matrix
classification_rep = None  # TODO: Generate classification report

print("Gene Expression Model Evaluation:")
# TODO: Add evaluation printing code

# TODO: Calculate additional biological metrics
# HINT: Extract true negatives, false positives, false negatives, true positives from confusion matrix
# HINT: Calculate sensitivity (TPR), specificity (TNR), and precision
tn, fp, fn, tp = 0, 0, 0, 0  # TODO: Extract confusion matrix values
sensitivity = 0.0  # TODO: Calculate sensitivity
specificity = 0.0  # TODO: Calculate specificity
precision = 0.0  # TODO: Calculate precision

print(f"\nBiological Metrics:")
print(f"  Sensitivity (True Positive Rate): {sensitivity:.3f}")
print(f"  Specificity (True Negative Rate): {specificity:.3f}")
print(f"  Precision: {precision:.3f}")

# ============================================================================
# [BONUS 15 pts] BONUS: Advanced Bayesian Network Analysis
# ============================================================================

print("\nBONUS: Advanced Bayesian Network Analysis")
print("-" * 50)

# TODO: Implement advanced network analysis function
# HINT: Use community detection and betweenness centrality
def advanced_network_analysis(graph, data, target_col):
    """
    Perform advanced network analysis
    
    Args:
        graph: NetworkX graph object
        data: DataFrame with features
        target_col: Name of target column
    
    Returns:
        dict: Dictionary of advanced analysis results
    """
    results = {}
    
    # TODO: Find communities
    # HINT: Use nx.community.greedy_modularity_communities(graph)
    communities = None  # TODO: Find communities
    results['num_communities'] = 0  # TODO: Count communities
    results['avg_community_size'] = 0.0  # TODO: Calculate average size
    
    # TODO: Calculate betweenness centrality for key nodes
    # HINT: Use nx.betweenness_centrality(graph)
    # HINT: Sort by centrality and take top 3
    betweenness = None  # TODO: Calculate betweenness
    top_betweenness = None  # TODO: Find top betweenness nodes
    results['top_betweenness'] = top_betweenness
    
    # TODO: Analyze correlation between network position and target
    # HINT: This is optional - you can skip this part
    results['node_target_correlations'] = []
    
    return results

# TODO: Perform advanced analysis for gene network
# HINT: Call advanced_network_analysis() with gene_graph
gene_advanced = None  # TODO: Perform advanced analysis
print("Advanced Gene Network Analysis:")
# TODO: Add advanced analysis printing code

# ============================================================================
# [CONCEPTUAL 15 pts] CONCEPTUAL QUESTIONS
# ============================================================================

print("\nCONCEPTUAL QUESTIONS")
print("-" * 50)

# TODO: Answer these conceptual questions in your code comments
"""
Q1: What is the main advantage of Bayesian Networks over other machine learning methods 
     in bioinformatics applications?

A) They can handle missing data better
B) They provide interpretable probabilistic relationships between variables
C) They are faster to train
D) They require less data

Your answer: [TODO: Choose A, B, C, or D]
"""

q1_answer = "TODO"  # TODO: Replace with your answer (A, B, C, or D)

"""
Q2: In the context of gene expression analysis, what does a high correlation between 
     two genes in a Bayesian Network typically indicate?

A) The genes are physically close on the chromosome
B) The genes may be co-regulated or functionally related
C) The genes have similar mutation rates
D) The genes are always expressed together

Your answer: [TODO: Choose A, B, C, or D]
"""

q2_answer = "TODO"  # TODO: Replace with your answer (A, B, C, or D)

"""
Q3: When analyzing protein interaction networks, what does a high betweenness centrality 
     of a protein typically suggest?

A) The protein is highly expressed
B) The protein acts as a hub or bridge in the network
C) The protein has many direct interactions
D) The protein is essential for cell survival

Your answer: [TODO: Choose A, B, C, or D]
"""

q3_answer = "TODO"  # TODO: Replace with your answer (A, B, C, or D)

print("Conceptual Questions Answered:")
print(f"Q1: {q1_answer}")
print(f"Q2: {q2_answer}")
print(f"Q3: {q3_answer}")

# ============================================================================
# FINAL RESULTS AND SUMMARY
# ============================================================================

print("\nFINAL RESULTS AND SUMMARY")
print("=" * 60)

# TODO: Store final results
# HINT: Create a dictionary with all your results
final_results = {
    'gene_network_edges': 0,  # TODO: Count gene network edges
    'disease_network_edges': 0,  # TODO: Count disease network edges
    'protein_network_nodes': 0,  # TODO: Count protein network nodes
    'protein_network_edges': 0,  # TODO: Count protein network edges
    'inference_accuracy': 0.0,  # TODO: Store inference accuracy
    'gene_network_density': 0.0,  # TODO: Store gene network density
    'disease_network_density': 0.0,  # TODO: Store disease network density
    'protein_network_density': 0.0,  # TODO: Store protein network density
    'q1_answer': q1_answer,
    'q2_answer': q2_answer,
    'q3_answer': q3_answer
}

print("Project Summary:")
print(f" Gene expression network: {final_results['gene_network_edges']} edges")
print(f" Disease markers network: {final_results['disease_network_edges']} edges")
print(f" Protein interaction network: {final_results['protein_network_nodes']} nodes, {final_results['protein_network_edges']} edges")
print(f" Inference accuracy: {final_results['inference_accuracy']:.3f}")
print(f" Network densities: Gene={final_results['gene_network_density']:.3f}, Disease={final_results['disease_network_density']:.3f}, Protein={final_results['protein_network_density']:.3f}")

print("\nBayesian Network Bioinformatics Project Completed!")

# ============================================================================
# SUBMISSION CHECKLIST
# ============================================================================

print("\n" + "=" * 60)
print("SUBMISSION CHECKLIST")
print("=" * 60)
print("✅ Make sure you have completed all TODO sections")
print("✅ Test your code to ensure it runs without errors")
print("✅ Answer all conceptual questions (Q1, Q2, Q3)")
print("✅ Implement all required functions")
print("✅ Calculate all required metrics and properties")
print("✅ Document your biological insights in comments")
print("=" * 60) 