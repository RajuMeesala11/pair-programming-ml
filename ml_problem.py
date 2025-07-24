"""
Machine Learning Problem for Shopify Pair Programming Interview

This file contains a machine learning problem that involves:
1. Loading and preprocessing a dataset
2. Exploring and visualizing the data
3. Building and training a model
4. Evaluating the model's performance
5. Making predictions

The problem focuses on a product recommendation system, which is relevant
to an e-commerce platform like Shopify.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA

class ProductRecommender:
    """
    A product recommendation system that suggests similar products based on features.
    This class demonstrates how to build a simple recommendation system using 
    machine learning techniques.
    """
    
    def __init__(self, n_neighbors=5):
        """
        Initialize the recommender with the number of neighbors to find.
        
        Args:
            n_neighbors (int): Number of similar products to recommend
        """
        self.n_neighbors = n_neighbors
        self.model = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='auto')
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)  # For visualization
        
    def generate_sample_data(self, n_samples=100, n_features=5):
        """
        Generate sample product data for demonstration.
        
        Args:
            n_samples (int): Number of products to generate
            n_features (int): Number of features for each product
            
        Returns:
            DataFrame: Sample product data
        """
        # Generate random feature values
        features = np.random.rand(n_samples, n_features)
        
        # Generate random product IDs and names
        product_ids = [f"P{i:04d}" for i in range(1, n_samples+1)]
        product_names = [f"Product {i}" for i in range(1, n_samples+1)]
        
        # Create price column with some correlation to features
        base_price = 10 + features[:, 0] * 90  # Price between $10 and $100
        
        # Create categories
        categories = np.random.choice(['Clothing', 'Electronics', 'Home', 'Beauty', 'Sports'], n_samples)
        
        # Create DataFrame
        data = pd.DataFrame(features, columns=[f'feature_{i+1}' for i in range(n_features)])
        data['product_id'] = product_ids
        data['product_name'] = product_names
        data['price'] = base_price
        data['category'] = categories
        
        return data
    
    def preprocess_data(self, data):
        """
        Preprocess the data by scaling features.
        
        Args:
            data (DataFrame): Product data
            
        Returns:
            ndarray: Scaled feature data
        """
        # Select numerical features
        feature_cols = [col for col in data.columns if col.startswith('feature_')]
        X = data[feature_cols].values
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled
    
    def fit(self, data):
        """
        Fit the recommendation model on the given data.
        
        Args:
            data (DataFrame): Product data
        """
        self.data = data
        X_scaled = self.preprocess_data(data)
        self.model.fit(X_scaled)
        
        # For visualization
        self.X_pca = self.pca.fit_transform(X_scaled)
    
    def get_recommendations(self, product_idx, visualize=False):
        """
        Get recommendations for a specific product.
        
        Args:
            product_idx (int): Index of the product in the dataset
            visualize (bool): Whether to visualize the recommendations
            
        Returns:
            DataFrame: Recommended products
        """
        # Get features for the query product
        feature_cols = [col for col in self.data.columns if col.startswith('feature_')]
        query_features = self.data.iloc[product_idx][feature_cols].values.reshape(1, -1)
        query_features_scaled = self.scaler.transform(query_features)
        
        # Find nearest neighbors
        distances, indices = self.model.kneighbors(query_features_scaled)
        
        # Skip the first result (which is the query product itself)
        recommendation_indices = indices[0][1:]
        
        # Get recommended products
        recommendations = self.data.iloc[recommendation_indices].copy()
        recommendations['similarity'] = 1 - distances[0][1:]  # Convert distance to similarity
        
        # Visualize if requested
        if visualize:
            self._visualize_recommendations(product_idx, recommendation_indices)
            
        return recommendations
    
    def _visualize_recommendations(self, query_idx, rec_indices):
        """
        Visualize the query product and its recommendations in 2D space.
        
        Args:
            query_idx (int): Index of the query product
            rec_indices (list): Indices of recommended products
        """
        plt.figure(figsize=(10, 8))
        
        # Plot all products
        plt.scatter(self.X_pca[:, 0], self.X_pca[:, 1], 
                   c='lightgray', alpha=0.5, label='All Products')
        
        # Plot query product
        plt.scatter(self.X_pca[query_idx, 0], self.X_pca[query_idx, 1], 
                   c='red', s=100, label='Query Product')
        
        # Plot recommended products
        plt.scatter(self.X_pca[rec_indices, 0], self.X_pca[rec_indices, 1], 
                   c='blue', s=100, alpha=0.7, label='Recommendations')
        
        plt.title('Product Recommendations Visualization')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def evaluate_recommendations(self, test_indices):
        """
        Evaluate the recommendation system using a subset of the data.
        
        Args:
            test_indices (list): Indices of test products
            
        Returns:
            dict: Evaluation metrics
        """
        # For demonstration, we'll calculate the average price difference
        # between query products and their recommendations
        price_diffs = []
        category_matches = []
        
        for idx in test_indices:
            query_price = self.data.iloc[idx]['price']
            query_category = self.data.iloc[idx]['category']
            
            recommendations = self.get_recommendations(idx)
            rec_prices = recommendations['price'].values
            rec_categories = recommendations['category'].values
            
            # Calculate price difference
            price_diff = np.abs(rec_prices - query_price).mean()
            price_diffs.append(price_diff)
            
            # Calculate category matches
            category_match = np.mean([1 if cat == query_category else 0 for cat in rec_categories])
            category_matches.append(category_match)
        
        return {
            'avg_price_difference': np.mean(price_diffs),
            'avg_category_match_rate': np.mean(category_matches)
        }


def main():
    """
    Main function to demonstrate the product recommender.
    """
    # Create recommender
    recommender = ProductRecommender(n_neighbors=5)
    
    # Generate sample data
    print("Generating sample product data...")
    data = recommender.generate_sample_data(n_samples=200, n_features=6)
    print(f"Generated {len(data)} products with {data.filter(like='feature').shape[1]} features.")
    
    # Display the first few rows
    print("\nSample of product data:")
    print(data.head())
    
    # Fit the model
    print("\nFitting recommendation model...")
    recommender.fit(data)
    
    # Get recommendations for a random product
    query_idx = np.random.randint(0, len(data))
    query_product = data.iloc[query_idx]
    
    print(f"\nQuery Product: {query_product['product_name']} (Category: {query_product['category']}, Price: ${query_product['price']:.2f})")
    
    recommendations = recommender.get_recommendations(query_idx, visualize=True)
    
    print("\nTop Recommendations:")
    for i, (_, rec) in enumerate(recommendations.iterrows(), 1):
        print(f"{i}. {rec['product_name']} - Category: {rec['category']}, Price: ${rec['price']:.2f}, Similarity: {rec['similarity']:.4f}")
    
    # Evaluate on a random subset
    print("\nEvaluating recommendation system...")
    test_indices = np.random.choice(len(data), size=20, replace=False)
    eval_metrics = recommender.evaluate_recommendations(test_indices)
    
    print("\nEvaluation Metrics:")
    print(f"Average Price Difference: ${eval_metrics['avg_price_difference']:.2f}")
    print(f"Average Category Match Rate: {eval_metrics['avg_category_match_rate']:.2%}")
    
    print("\nNext steps could include:")
    print("1. Load real product data instead of generated data")
    print("2. Improve features with more product attributes")
    print("3. Experiment with different algorithms like collaborative filtering")
    print("4. Implement user preference weighting")
    print("5. Build an API to serve recommendations")


if __name__ == "__main__":
    main()
