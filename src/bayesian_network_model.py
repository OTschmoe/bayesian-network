"""
Bayesian Network Model Implementation
Provides core Bayesian Network functionality for bioinformatics applications
"""

import pandas as pd
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class BayesianNetwork:
    """
    Bayesian Network implementation for bioinformatics applications
    """
    
    def __init__(self, structure_learning_method='correlation', threshold=0.3):
        """
        Initialize Bayesian Network
        
        Args:
            structure_learning_method: Method for learning network structure
            threshold: Correlation threshold for edge inclusion
        """
        self.structure_learning_method = structure_learning_method
        self.threshold = threshold
        self.graph = nx.DiGraph()
        self.conditional_probabilities = {}
        self.prior_probabilities = {}
        self.node_states = {}
        
    def learn_structure(self, data, target_col=None):
        """
        Learn Bayesian network structure from data
        
        Args:
            data: Input dataset
            target_col: Target column name
        """
        if self.structure_learning_method == 'correlation':
            return self._learn_structure_correlation(data, target_col)
        elif self.structure_learning_method == 'mutual_info':
            return self._learn_structure_mutual_info(data, target_col)
        else:
            raise ValueError(f"Unknown structure learning method: {self.structure_learning_method}")
    
    def _learn_structure_correlation(self, data, target_col=None):
        """
        Learn structure using correlation-based approach
        """
        # Calculate correlation matrix
        if target_col:
            features = data.drop(columns=[target_col])
            target = data[target_col]
            
            # Calculate correlations with target
            correlations = features.corrwith(target).abs()
            
            # Add edges from features to target
            for feature in features.columns:
                if correlations[feature] > self.threshold:
                    self.graph.add_edge(feature, target_col, weight=correlations[feature])
        
        # Calculate feature-feature correlations
        corr_matrix = data.corr().abs()
        
        # Add edges between features
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > self.threshold:
                    self.graph.add_edge(corr_matrix.columns[i], corr_matrix.columns[j], 
                                      weight=corr_matrix.iloc[i, j])
        
        return self.graph
    
    def _learn_structure_mutual_info(self, data, target_col=None):
        """
        Learn structure using mutual information approach
        """
        if target_col:
            features = data.drop(columns=[target_col])
            target = data[target_col]
            
            # Calculate mutual information with target
            for feature in features.columns:
                mi = self._calculate_mutual_information(data[feature], target)
                if mi > self.threshold:
                    self.graph.add_edge(feature, target_col, weight=mi)
        
        # Calculate mutual information between features
        for i, col1 in enumerate(data.columns):
            for j, col2 in enumerate(data.columns[i+1:], i+1):
                mi = self._calculate_mutual_information(data[col1], data[col2])
                if mi > self.threshold:
                    self.graph.add_edge(col1, col2, weight=mi)
        
        return self.graph
    
    def _calculate_mutual_information(self, x, y):
        """
        Calculate mutual information between two variables
        """
        # Create contingency table
        contingency = pd.crosstab(x, y)
        total = contingency.sum().sum()
        
        mi = 0
        for i in range(len(contingency.index)):
            for j in range(len(contingency.columns)):
                p_ij = contingency.iloc[i, j] / total
                p_i = contingency.iloc[i, :].sum() / total
                p_j = contingency.iloc[:, j].sum() / total
                
                if p_ij > 0 and p_i > 0 and p_j > 0:
                    mi += p_ij * np.log2(p_ij / (p_i * p_j))
        
        return mi
    
    def learn_parameters(self, data, target_col=None):
        """
        Learn conditional probability parameters from data
        
        Args:
            data: Input dataset
            target_col: Target column name
        """
        if target_col:
            # Learn prior probabilities for target
            target_counts = data[target_col].value_counts()
            total = len(data)
            for value in target_counts.index:
                self.prior_probabilities[target_col] = target_counts[value] / total
        
        # Learn conditional probabilities for each node
        for node in self.graph.nodes():
            self._learn_node_parameters(data, node)
    
    def _learn_node_parameters(self, data, node):
        """
        Learn conditional probability parameters for a specific node
        """
        parents = list(self.graph.predecessors(node))
        
        if not parents:
            # No parents - learn prior probability
            node_counts = data[node].value_counts()
            total = len(data)
            for value in node_counts.index:
                prob_key = f"P({node}={value})"
                self.conditional_probabilities[prob_key] = node_counts[value] / total
        else:
            # Has parents - learn conditional probabilities
            for parent in parents:
                self._learn_conditional_probabilities(data, node, parent)
    
    def _learn_conditional_probabilities(self, data, child, parent):
        """
        Learn conditional probabilities P(child|parent)
        """
        for parent_val in data[parent].unique():
            for child_val in data[child].unique():
                # Count instances where parent=parent_val and child=child_val
                numerator = len(data[(data[parent] == parent_val) & (data[child] == child_val)])
                
                # Count instances where parent=parent_val
                denominator = len(data[data[parent] == parent_val])
                
                if denominator > 0:
                    prob_key = f"P({child}={child_val}|{parent}={parent_val})"
                    self.conditional_probabilities[prob_key] = numerator / denominator
    
    def predict(self, features, target_col):
        """
        Perform inference to predict target variable
        
        Args:
            features: Input features
            target_col: Target column name
            
        Returns:
            Predictions and probabilities
        """
        predictions = []
        probabilities = []
        
        for _, row in features.iterrows():
            # Calculate posterior probability for each target value
            target_probs = {}
            
            for target_val in [0, 1]:  # Binary classification
                # Start with prior probability
                prior = self.prior_probabilities.get(f"{target_col}={target_val}", 0.5)
                likelihood = 1.0
                
                # Calculate likelihood using conditional probabilities
                for feature in features.columns:
                    feature_val = row[feature]
                    prob_key = f"P({feature}={feature_val}|{target_col}={target_val})"
                    
                    if prob_key in self.conditional_probabilities:
                        likelihood *= self.conditional_probabilities[prob_key]
                    else:
                        # Use default probability if not found
                        likelihood *= 0.5
                
                posterior = prior * likelihood
                target_probs[target_val] = posterior
            
            # Normalize probabilities
            total_prob = sum(target_probs.values())
            if total_prob > 0:
                for val in target_probs:
                    target_probs[val] /= total_prob
            
            # Make prediction
            predicted_class = max(target_probs, key=target_probs.get)
            predictions.append(predicted_class)
            probabilities.append(target_probs)
        
        return predictions, probabilities
    
    def evaluate(self, predictions, true_values):
        """
        Evaluate model performance
        
        Args:
            predictions: Model predictions
            true_values: True target values
            
        Returns:
            Dictionary of performance metrics
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
    
    def visualize_network(self, title="Bayesian Network", figsize=(12, 8)):
        """
        Visualize the Bayesian network structure
        """
        plt.figure(figsize=figsize)
        
        # Use hierarchical layout for directed graph
        pos = nx.spring_layout(self.graph, k=2, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, node_color='lightblue', 
                              node_size=1000, alpha=0.7)
        
        # Draw edges
        nx.draw_networkx_edges(self.graph, pos, edge_color='gray', 
                              arrows=True, arrowsize=20, alpha=0.6)
        
        # Draw labels
        nx.draw_networkx_labels(self.graph, pos, font_size=10, font_weight='bold')
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def get_network_properties(self):
        """
        Get network topology properties
        """
        if self.graph.number_of_nodes() == 0:
            return {}
        
        properties = {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'avg_degree': sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes(),
            'max_degree': max(dict(self.graph.degree()).values()),
            'is_directed': self.graph.is_directed(),
            'is_dag': nx.is_directed_acyclic_graph(self.graph)
        }
        
        return properties

class NaiveBayesClassifier:
    """
    Naive Bayes classifier implementation for bioinformatics
    """
    
    def __init__(self, smoothing=1.0):
        """
        Initialize Naive Bayes classifier
        
        Args:
            smoothing: Laplace smoothing parameter
        """
        self.smoothing = smoothing
        self.class_priors = {}
        self.feature_probs = {}
        self.classes = None
    
    def fit(self, X, y):
        """
        Train the Naive Bayes classifier
        
        Args:
            X: Feature matrix
            y: Target vector
        """
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        
        # Calculate class priors
        for class_val in self.classes:
            class_count = np.sum(y == class_val)
            self.class_priors[class_val] = (class_count + self.smoothing) / (n_samples + len(self.classes) * self.smoothing)
        
        # Calculate feature probabilities for each class
        for feature_idx in range(n_features):
            self.feature_probs[feature_idx] = {}
            
            for class_val in self.classes:
                class_mask = y == class_val
                class_samples = X[class_mask, feature_idx]
                
                # Calculate probability for each feature value
                unique_values, counts = np.unique(class_samples, return_counts=True)
                
                for value, count in zip(unique_values, counts):
                    prob_key = (feature_idx, class_val, value)
                    self.feature_probs[feature_idx][prob_key] = (count + self.smoothing) / (len(class_samples) + len(unique_values) * self.smoothing)
    
    def predict(self, X):
        """
        Predict class labels
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted class labels
        """
        predictions = []
        
        for sample in X:
            class_scores = {}
            
            for class_val in self.classes:
                # Start with class prior
                score = np.log(self.class_priors[class_val])
                
                # Add log probability of features
                for feature_idx, feature_val in enumerate(sample):
                    prob_key = (feature_idx, class_val, feature_val)
                    
                    if prob_key in self.feature_probs[feature_idx]:
                        score += np.log(self.feature_probs[feature_idx][prob_key])
                    else:
                        # Use smoothing for unseen values
                        score += np.log(self.smoothing / (len(self.feature_probs[feature_idx]) + self.smoothing))
                
                class_scores[class_val] = score
            
            # Predict class with highest score
            predicted_class = max(class_scores, key=class_scores.get)
            predictions.append(predicted_class)
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        
        Args:
            X: Feature matrix
            
        Returns:
            Class probabilities
        """
        probas = []
        
        for sample in X:
            class_scores = {}
            
            for class_val in self.classes:
                # Calculate unnormalized scores
                score = np.log(self.class_priors[class_val])
                
                for feature_idx, feature_val in enumerate(sample):
                    prob_key = (feature_idx, class_val, feature_val)
                    
                    if prob_key in self.feature_probs[feature_idx]:
                        score += np.log(self.feature_probs[feature_idx][prob_key])
                    else:
                        score += np.log(self.smoothing / (len(self.feature_probs[feature_idx]) + self.smoothing))
                
                class_scores[class_val] = score
            
            # Convert to probabilities
            max_score = max(class_scores.values())
            exp_scores = {cls: np.exp(score - max_score) for cls, score in class_scores.items()}
            total = sum(exp_scores.values())
            
            sample_probas = [exp_scores[cls] / total for cls in self.classes]
            probas.append(sample_probas)
        
        return np.array(probas)

def create_bayesian_network_from_data(data, target_col, method='correlation', threshold=0.3):
    """
    Create and train a Bayesian network from data
    
    Args:
        data: Input dataset
        target_col: Target column name
        method: Structure learning method
        threshold: Correlation threshold
        
    Returns:
        Trained Bayesian network
    """
    # Create network
    bn = BayesianNetwork(structure_learning_method=method, threshold=threshold)
    
    # Learn structure
    bn.learn_structure(data, target_col)
    
    # Learn parameters
    bn.learn_parameters(data, target_col)
    
    return bn

def evaluate_bayesian_network(bn, X_test, y_test):
    """
    Evaluate Bayesian network performance
    
    Args:
        bn: Trained Bayesian network
        X_test: Test features
        y_test: Test targets
        
    Returns:
        Performance metrics
    """
    predictions, _ = bn.predict(X_test, y_test.name if hasattr(y_test, 'name') else 'target')
    return bn.evaluate(predictions, y_test) 