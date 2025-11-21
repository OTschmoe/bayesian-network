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

# HINT: Use pd.read_csv() to load the CSV files from the data/ folder
gene_data = pd.read_csv("data/gene_expression.csv") 
disease_data = pd.read_csv("data/disease_markers.csv")  
protein_data = pd.read_csv("data/protein_interactions.csv")

# HINT: Use .shape, .columns, and .value_counts() to explore the data
#Print the shapes of all the datasets
print("Data Shapes")
print(f"Gene Expression Dataset Shape: {gene_data.shape if gene_data is not None else 'Not loaded'}")
print(f"Disease Markers Dataset Shape: {disease_data.shape if disease_data is not None else 'Not loaded'}")
print(f"Protein Interactions Dataset Shape: {protein_data.shape if protein_data is not None else 'Not loaded'}")

#print the first few column names for each dataset
print("\nData Column Names")
print("First 10 Gene Expression Dataset column names:")
print("    ", end="")
for i in range(10):
    print(f"{gene_data.columns[i]}, ", end="")
print()
print()

print("First 10 Disease Markers Dataset column names:")
print("    ", end="")
for i in range(10):
    print(f"{disease_data.columns[i]}, ", end="")
print()
print()

print("First 5 Protein Interactions Dataset column names:")
print("    ", end="")
for i in range(5):
    print(f"{protein_data.columns[i]}, ", end="")
print()
print()

#print the number of unique values in each dataset
print("Data Values")
print(f"Number of Unique Gene Expression Dataset values: {len(gene_data.value_counts()) if gene_data is not None else 'Not loaded'}")
print(f"Number of Unique Disease Markers Dataset values: {len(disease_data.value_counts()) if disease_data is not None else 'Not loaded'}")
print(f"Number of Unique Protein Interactions Dataset values: {len(protein_data.value_counts()) if gene_data is not None else 'Not loaded'}")

gene_stats = gene_data.describe()  
print(f"\nGene Expression Statistics:")
print(f"{gene_stats}")

#calculate top correlated genes
genes_cut = gene_data.drop(columns=["sample_id", "disease_status"]) #remove the sample id and disease status column, so it's just the gene columns left.
gene_correlations = genes_cut.corrwith(gene_data["disease_status"]) #now correlate with the genes to disease_status only.
top_correlated_genes = gene_correlations.abs().nlargest(5)
print(f"\nTop 5 genes correlated with disease status:")
for i, gene_name in enumerate(top_correlated_genes.index, start=1): #this makes it go 1-5 with the gene name
    corr_value = gene_correlations[gene_name] #this grabs the correlation for the gene name
    print(i, "  ", gene_name, "  ", corr_value) #prints it all.

# ============================================================================
# [15 pts] STEP 2: Data Preprocessing and Feature Engineering
# ============================================================================

print("\nSTEP 2: Data Preprocessing and Feature Engineering")
print("-" * 50)

gene_features = gene_data.drop(columns=["sample_id", "disease_status"]) #once again drop the sample_id and disease_status to ONLY get the genes.
gene_target = gene_data["disease_status"]  #the target is disease_status i think.

scaler = StandardScaler()  #Create scaler
gene_features_scaled = pd.DataFrame(scaler.fit_transform(gene_features), columns=gene_features.columns) #we make a pandas dataframe object and use the scaler fit transform to scale everything, then say columns = gene_features.columns to keep the same names.

#this creates binary somehow
gene_features_binary = (gene_features_scaled > 0).astype(int)  # Create binary features
gene_features_binary.columns = [col + "_high" for col in gene_features_binary.columns]  #this renames all the columns and adds _high

gene_features_combined = pd.concat([gene_features_scaled, gene_features_binary])  # concat scaled features and binary features
print(f"Combined feature set shape: {gene_features_combined.shape if gene_features_combined is not None else 'Not implemented'}")

disease_features = disease_data.drop(columns=['patient_id', 'diabetes_status']) # drop columns to get the data from the features
disease_target = disease_data['diabetes_status'] 

# HINT: Find SNP columns using list comprehension with .startswith('rs')
# NOTE: The dataset now contains real SNP names like rs7903146, rs12255372, etc.
snp_columns = [col for col in disease_features.columns if col.startswith('rs')] # any columns that starts with 'rs' to find the dataset
clinical_columns = ['age', 'bmi', 'glucose', 'insulin', 'hdl_cholesterol'] #other columns

#Creating snp interactions
snp_interactions = pd.DataFrame()  #this needs to be a dataframe to concatinate it later

for i in range(len(snp_columns)):
    for j in range(i+1, len(snp_columns)):
        col1 = snp_columns[i]
        col2 = snp_columns[j]
        new_col = str(col1) + "by" + str(col2)
        snp_interactions[new_col] = disease_features[col1] * disease_features[col2]

#this concatonates the original and the new one.
disease_features_combined = pd.concat([disease_features, snp_interactions], axis=1)  #Combine disease features
print(f"Disease features combined shape: {disease_features_combined.shape if disease_features_combined is not None else 'Not implemented'}")


# ============================================================================
# [18 pts] STEP 3: Bayesian Network Structure Learning - Zamzam
# ============================================================================

print("\nSTEP 3: Bayesian Network Structure Learning")
print("-" * 50)

# Calculate absolute correlations and find edges above threshold
def learn_bayesian_structure(data, threshold=0.3):
    """
    Learn Bayesian network structure using correlation-based approach
    
    Args:
        data: DataFrame with features
        threshold: Correlation threshold for creating edges
    
    Returns:
        list: List of tuples (node1, node2, correlation)
    """
    # Using data.corr().abs() to get absolute correlations
    correlations = data.corr().abs() #Calculate correlations
    edges = []
    
    #Using nested loops to check each pair of features and only adding the edges if correlation is bigger
    cols = correlations.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            node1 = cols[i]
            node2 = cols[j]
            corr_val = correlations.iloc[i, j]
            if corr_val > threshold:
                edges.append((node1, node2, corr_val))
    
    return edges

# Model is learning the structure for gene expression network using a subset of the genes
selected_genes = ['CDK1', 'MAPK1', 'PIK3CA', 'BCL2', 'GLUT1', 'MYC']
gene_subset = gene_features_combined[selected_genes] 
gene_network_edges = learn_bayesian_structure(gene_subset, threshold=0.3)  # this is where it learns it's gene network structure
print(f"Gene network edges found: {len(gene_network_edges) if gene_network_edges is not None else 'Not implemented'}")

# HINT: Using the first 15 columns of disease_features_combined learn the structure for the disease markers network
disease_subset = disease_features_combined.iloc[:, :15]
disease_network_edges = learn_bayesian_structure(disease_subset, threshold=0.1)  # TODO: Learn disease network structure
print(f"Disease network edges found: {len(disease_network_edges) if disease_network_edges is not None else 'Not implemented'}")

# Creating network graph
gene_graph = nx.Graph()
for n1, n2, w in gene_network_edges:
    gene_graph.add_edge(n1, n2, weight=w)
 
disease_graph =  nx.Graph() #Create disease graph
for n1, n2, w in disease_network_edges:
    disease_graph.add_edge(n1, n2, weight=w)

print(f"Gene network nodes: {gene_graph.number_of_nodes() if gene_graph is not None else 'Not created'}, edges: {gene_graph.number_of_edges() if gene_graph is not None else 'Not created'}")
print(f"Disease network nodes: {disease_graph.number_of_nodes() if disease_graph is not None else 'Not created'}, edges: {disease_graph.number_of_edges() if disease_graph is not None else 'Not created'}")

# ============================================================================
# [15 pts] STEP 4: Conditional Probability Calculations - NATHEN
# ============================================================================

print("\nSTEP 4: Conditional Probability Calculations")
print("-" * 50)

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
    
    # Loop through each feature
    for feature in feature_cols:
        if data[feature].dtype in ['int64', 'float64', 'bool']:
            continue

        # HINT: Use len() and boolean indexing to count occurrences
        # HINT: Calculate P(target=val1|feature=val2) for all combinations
        for feature_val in [0, 1]:
            for target_val in [0, 1]:
                # HINT: Count samples where both conditions are met
                # HINT: Divide by count of samples where feature condition is met

                feature_count = len(data[data[feature] == feature_val]) # Count how many samples have feature = feature_val
                joint_count = len(data[(data[feature] == feature_val) & (data[target_col] == target_val)]) # Count how many have BOTH feature = feature_val AND target = target_val
                prob = joint_count / feature_count if feature_count > 0 else 0.0 # Conditional probability P(target_val | feature_val)

                conditional_probs[f"P({target_col}={target_val}|{feature}={feature_val})"] = prob
    
    return conditional_probs

# HINT: Use first 5 binary features
gene_binary_features = gene_features_binary.iloc[:, :5]  # Select binary features

# Calculate conditional probabilities
gene_conditional_probs = calculate_conditional_probabilities(
    pd.concat([gene_binary_features, gene_target], axis=1),
    target_col="disease_status",
    feature_cols=gene_binary_features.columns
)

print("Conditional probabilities for gene expression:")
# print out probabilities
for key, value in gene_conditional_probs.items():
    print(f"{key}: {value:.2f}")

# HINT: Ensure SNP columns are binary (0/1) before calculation
# HINT: Use first 5 SNP columns from snp_columns
disease_binary_features = disease_data[snp_columns[:5]].copy()  # Select disease binary features

# HINT: Check if values are binary, if not convert to binary
# HINT: Use (disease_binary_features[col] > 0).astype(int) to binarize
print("\nStep 4c: Checking SNP columns for binary values:")
# Add SNP checking code
for col in disease_binary_features.columns:
    unique_vals = sorted(disease_binary_features[col].unique())
    print(f"{col} unique values: {unique_vals}")

    # Binarize if not already 0/1
    if set(unique_vals) != {0, 1}:
        disease_binary_features[col] = (disease_binary_features[col] > 0).astype(int)

disease_data_combined = pd.concat([disease_binary_features, disease_target], axis=1) # Combine disease data
# Calculate disease conditional probabilities
disease_conditional_probs = calculate_conditional_probabilities(
    disease_data_combined,
    target_col="diabetes_status",
    feature_cols=disease_binary_features.columns
)

print(f"Disease binary features shape: {disease_binary_features.shape if disease_binary_features is not None else 'Not implemented'}")
print(f"Disease target shape: {disease_target.shape if disease_target is not None else 'Not implemented'}")
print(f"SNP columns: {snp_columns[:5] if snp_columns is not None else 'Not implemented'}")

print("\nConditional probabilities for disease markers:")
# Add disease probability printing code
for key, value in disease_conditional_probs.items():
    print(f"{key}: {value:.4f}")

# ============================================================================
# [18 pts] STEP 5: Probabilistic Inference - Owen
# ============================================================================

# print("\nSTEP 5: Probabilistic Inference")
# print("-" * 50)

# # TODO: Implement Naive Bayes inference function
# # HINT: Use the conditional probabilities and prior probabilities
# # HINT: Calculate likelihood for each class and apply Bayes' theorem
# def naive_bayes_inference(features, conditional_probs, prior_probs):
#     """
#     Perform naive Bayes inference
    
#     Args:
#         features: DataFrame of features
#         conditional_probs: Dictionary of conditional probabilities
#         prior_probs: List of prior probabilities [P(class=0), P(class=1)]
    
#     Returns:
#         list: List of predicted classes
#     """
#     predictions = []
    
#     #loop through each sample
#     for _, row in features.iterrows():
#         #calculate likelihood for each class
#         likelihood_0 = 1.0
#         likelihood_1 = 1.0
        
#         for feature in features.columns:
#             feature_val = row[feature]
            
#             #get conditional probabilities from dictionary
#             prob_0 = conditional_probs[feature][0].get(feature_val, 0.5) #use the feature with class 0 and the default value of .5
#             prob_1 = conditional_probs[feature][1].get(feature_val, 0.5)
            
#             likelihood_0 *= prob_0  #update likelihood_0
#             likelihood_1 *= prob_1  #update likelihood_1
        
#         #apply prior probabilities
#         posterior_0 = likelihood_0 * prior_probs[0]
#         posterior_1 = likelihood_1 * prior_probs[1]
        
#         #normalize probabilities
#         total = posterior_0 + posterior_1  #total for posteriors
#         posterior_0 /= total  #normalize by dividing posterior / total
#         posterior_1 /= total 
        
#         #choose class with higher posterior probability
#         #class in this case is either 0 or 1, so we choose the one thats bigger.
#         predictions.append(0 if posterior_0 > posterior_1 else 1) 
    
#     return predictions

# #train_test_split returns a tuple, so we assign each of these to a value in that tuple.
# X_train, X_test, y_train, y_test = train_test_split(disease_features_combined, disease_target, test_size=.3, random_state=42)  #I think this is based on disease stuff??

# # calculate prior probabilities
# total = len(y_train)

# prior_probs = [
#     y_train.value_counts().get(0, 0) / total,  #probability of 0
#     y_train.value_counts().get(1, 0) / total   #probability of 1
# ]

# #perform inference
# predictions = naive_bayes_inference(X_test, disease_conditional_probs, prior_probs)

# # calculate accuracy
# accuracy = accuracy_score(y_test, predictions)
# print(f"Naive Bayes inference accuracy: {accuracy:.3f}")

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
    
    # Calculating basic properties
    properties['nodes'] = graph.number_of_nodes() #Count nodes
    properties['edges'] = graph.number_of_edges()  #Count edges
    properties['density'] = nx.density(graph)  #Calculate density
    
    # Calculate centrality measures
    if graph.number_of_nodes() > 0:
        degree_dict = dict(graph.degree())
        properties['avg_degree'] = sum(degree_dict.values()) / graph.number_of_nodes() # TODO: Calculate average degree
        
        #Calculating maximum degree
        properties['max_degree'] =  max(degree_dict.values()) #Calculate max degree 
        # HINT: Use nx.average_clustering(graph)
        properties['avg_clustering'] =nx.average_clustering(graph) #Calculate clustering
        
        # TODO: Calculate number of connected components
        # HINT: Use nx.number_connected_components(graph)
        if nx.is_directed(graph):
            properties['connected_components'] = nx.number_weakly_connected_components(graph)
        else:
            properties['connected_components'] = nx.number_connected_components(graph)
    else:
        properties['avg_degree'] = 0.0
        properties['max_degree'] = 0
        properties['avg_clustering'] = 0.0
        properties['connected_components'] = 0
    
    return properties

# TODO: Analyze gene network
# HINT: Call analyze_network_properties() with gene_graph
gene_properties = analyze_network_properties(gene_graph)
print("Gene Network Properties:")
for key, value in gene_properties.items():
    print(f"  {key}: {value}")

#Analyze disease network
disease_properties = analyze_network_properties(disease_graph)
print("\nDisease Network Properties:")
for key, value in disease_properties.items():
    print(f"  {key}: {value}")

# Visualize networks
print("\nVisualizing networks...")

# Gene network
visualize_network(gene_graph, title="Gene Expression Network")
plot_network_metrics(gene_graph, title="Gene Network Metrics")

# Disease network
visualize_network(disease_graph, title="Disease Marker Network")
plot_network_metrics(disease_graph, title="Disease Network Metrics")

# ============================================================================
# [10 pts] STEP 7: Protein Interaction Network Analysis - Owen
# ============================================================================

print("\nSTEP 7: Protein Interaction Network Analysis")
print("-" * 50)

# create protein interaction network
protein_graph = nx.Graph() 

# add edges from protein interaction data
for index, row in protein_data.iterrows():
    protein_graph.add_edge(row["protein_1"], row["protein_2"], weight = row["interaction_score"]) #use the protein1, protein2, and interaction score columns to add edges.

#analyze protein network
protein_properties = analyze_network_properties(protein_graph)
print("Protein Interaction Network Properties:")
for key, value in protein_properties.items():
    print("  ", key, value)

# find hub proteins (high degree nodes)
protein_degrees = dict(protein_graph.degree())  # Get protein degrees
hub_proteins = [(degree, protein) for protein, degree in protein_graph.degree()]  #create list
hub_proteins.sort(reverse=True) #reverse so biggest first
print(f"\nTop 5 hub proteins:")
for degree, protein in hub_proteins[:5]: #only print the first 5
    print("    ", protein, degree)

# Analyze interaction types
interaction_types = protein_data["interaction_type"].value_counts()  # this gives the counts for each type of interaction
print(f"\nInteraction type distribution:")
for interaction, count in interaction_types.items():
    print("   ", interaction, count)

# Visualize protein network
print("\nVisualizing protein interaction network...")
visualize_network(protein_graph, title="Protein Interaction Network")
plot_network_metrics(protein_graph, title="Protein Network Metrics")

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

q1_answer = "B"  # TODO: Replace with your answer (A, B, C, or D)

"""
Q2: In the context of gene expression analysis, what does a high correlation between 
     two genes in a Bayesian Network typically indicate?

A) The genes are physically close on the chromosome
B) The genes may be co-regulated or functionally related
C) The genes have similar mutation rates
D) The genes are always expressed together

Your answer: [TODO: Choose A, B, C, or D]
"""

q2_answer = "B"  # TODO: Replace with your answer (A, B, C, or D)

"""
Q3: When analyzing protein interaction networks, what does a high betweenness centrality 
     of a protein typically suggest?

A) The protein is highly expressed
B) The protein acts as a hub or bridge in the network
C) The protein has many direct interactions
D) The protein is essential for cell survival

Your answer: [TODO: Choose A, B, C, or D]
"""

q3_answer = "B"  # TODO: Replace with your answer (A, B, C, or D)

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