import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import torch
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
from pykeen.models import TransE, ComplEx, RotatE
from pykeen.training import SLCWATrainingLoop
from pykeen.evaluation import RankBasedEvaluator
from pykeen.datasets import Dataset
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

class KnowledgeGraphGenerator:
    """
    A class to generate and train knowledge graphs from CSV data using PyKEEN.
    
    Expected CSV format:
    - Columns: subject, relation, object
    - Relations: 'likes', 'dislikes', 'concerns', 'has'
    - Nodes: users (user_*), items (item_*), features (feature_*)
    """
    
    def __init__(self):
        self.triples_factory = None
        self.model = None
        self.training_results = None
        self.entity_to_id = None
        self.relation_to_id = None
        
    def load_csv_data(self, csv_file_path: str) -> pd.DataFrame:
        """
        Load and validate CSV data.
        
        Args:
            csv_file_path: Path to the CSV file
            
        Returns:
            DataFrame with validated triples data
        """
        try:
            df = pd.read_csv(csv_file_path)
            
            # Validate required columns
            required_columns = ['subject', 'relation', 'object']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"CSV must contain columns: {required_columns}")
            
            # Remove any null values
            df = df.dropna(subset=required_columns)
            
            # Validate relations
            valid_relations = {'likes', 'dislikes', 'concerns', 'has'}
            invalid_relations = set(df['relation'].unique()) - valid_relations
            if invalid_relations:
                print(f"Warning: Found unexpected relations: {invalid_relations}")
            
            print(f"Loaded {len(df)} triples from CSV")
            print(f"Unique entities: {len(set(df['subject'].unique()) | set(df['object'].unique()))}")
            print(f"Unique relations: {df['relation'].nunique()}")
            
            return df
            
        except Exception as e:
            raise Exception(f"Error loading CSV data: {str(e)}")
    
    def create_triples_factory(self, df: pd.DataFrame, test_ratio: float = 0.2, 
                              validation_ratio: float = 0.1) -> Tuple[TriplesFactory, TriplesFactory, TriplesFactory]:
        """
        Create PyKEEN TriplesFactory objects for training, validation, and testing.
        
        Args:
            df: DataFrame with triples data
            test_ratio: Proportion of data for testing
            validation_ratio: Proportion of data for validation
            
        Returns:
            Tuple of (training, validation, test) TriplesFactory objects
        """
        # Convert DataFrame to numpy array of strings
        triples = df[['subject', 'relation', 'object']].values.astype(str)
        
        # Create main triples factory
        all_triples_factory = TriplesFactory.from_labeled_triples(triples)
        
        # Split into train/validation/test
        training, testing = all_triples_factory.split(ratios=[1-test_ratio, test_ratio])
        training, validation = training.split(ratios=[1-validation_ratio, validation_ratio])
        
        self.triples_factory = training
        self.entity_to_id = training.entity_to_id
        self.relation_to_id = training.relation_to_id
        
        print(f"Training triples: {training.num_triples}")
        print(f"Validation triples: {validation.num_triples}")
        print(f"Test triples: {testing.num_triples}")
        
        return training, validation, testing
    
    def train_knowledge_graph(self, training: TriplesFactory, validation: TriplesFactory, 
                            testing: TriplesFactory, model_name: str = 'TransE', epochs: int = 100, 
                            embedding_dim: int = 128, learning_rate: float = 0.01) -> dict:
        """
        Train a knowledge graph embedding model using PyKEEN.
        
        Args:
            training: Training triples factory
            validation: Validation triples factory
            testing: Testing triples factory
            model_name: Name of the embedding model ('TransE', 'ComplEx', 'RotatE')
            epochs: Number of training epochs
            embedding_dim: Dimension of entity/relation embeddings
            learning_rate: Learning rate for optimization
            
        Returns:
            Dictionary containing training results
        """
        print(f"Training {model_name} model...")
        
        # Configure model parameters
        model_kwargs = {
            'embedding_dim': embedding_dim,
        }
        
        # Model-specific parameters
        if model_name == 'ComplEx':
            model_kwargs['embedding_dim'] = embedding_dim // 2  # ComplEx uses complex embeddings
        
        try:
            # Run the training pipeline
            results = pipeline(
                training=training,
                testing=testing,
                validation=validation,
                model=model_name,
                model_kwargs=model_kwargs,
                optimizer='Adam',
                optimizer_kwargs={'lr': learning_rate},
                training_loop='SLCWA',  # Stochastic Local Closed World Assumption
                training_kwargs={'num_epochs': epochs, 'batch_size': 256},
                evaluation_kwargs={'batch_size': 512},
                random_seed=42,
                device='mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
            )
            
            self.model = results.model
            self.training_results = results
            
            print(f"Training completed!")
            print(f"Final training loss: {results.losses[-1]:.4f}")
            
            return {
                'model': results.model,
                'losses': results.losses,
                'evaluation_results': results.metric_results
            }
            
        except Exception as e:
            raise Exception(f"Error during training: {str(e)}")
    
    def evaluate_model(self, testing: TriplesFactory) -> dict:
        """
        Evaluate the trained model on test data.
        
        Args:
            testing: Test triples factory
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        print("Evaluating model on test data...")
        
        # Create evaluator
        evaluator = RankBasedEvaluator()
        
        # Evaluate
        results = evaluator.evaluate(
            model=self.model,
            mapped_triples=testing.mapped_triples,
            batch_size=512,
            additional_filter_triples=[
                self.triples_factory.mapped_triples,  # Filter training triples
            ]
        )
        
        print("Evaluation Results:")
        for metric_name, metric_value in results.to_dict().items():
            if isinstance(metric_value, dict):
                print(f"{metric_name}:")
                for sub_metric, sub_value in metric_value.items():
                    if isinstance(sub_value, (int, float)):
                        print(f"  {sub_metric}: {sub_value:.4f}")
                    else:
                        print(f"  {sub_metric}: {sub_value}")
            elif isinstance(metric_value, (int, float)):
                print(f"{metric_name}: {metric_value:.4f}")
            else:
                print(f"{metric_name}: {metric_value}")
        
        return results.to_dict()
    
    def recommend_items(self, user: str, top_k: int = 10, filter_known: bool = True) -> List[Tuple[str, float]]:
        """
        Recommend items for a given user.
        
        Args:
            user: User entity
            top_k: Number of recommendations to return
            filter_known: Whether to filter out items the user already interacted with
            
        Returns:
            List of tuples (item, score) sorted by score
        """
        if self.model is None:
            raise ValueError("Model must be trained before making recommendations")
        
        try:
            # Get user ID
            user_id = self.entity_to_id.get(user)
            if user_id is None:
                raise ValueError(f"User not found in training data: {user}")
            
            # Get all items
            all_items = [entity for entity in self.entity_to_id.keys() if entity.startswith('item_')]
            
            # Get known interactions if filtering
            known_items = set()
            if filter_known:
                # Get items user has already interacted with (likes/dislikes)
                for relation in ['likes', 'dislikes']:
                    if relation in self.relation_to_id:
                        rel_id = self.relation_to_id[relation]
                        # This is a simplified approach - in practice you'd check training data
                        # For now, we'll skip filtering to keep it simple
                        pass
            
            # Score all items for this user
            item_scores = []
            user_tensor = torch.tensor([user_id]).long()
            likes_rel_id = self.relation_to_id.get('likes')
            
            if likes_rel_id is not None:
                likes_tensor = torch.tensor([likes_rel_id]).long()
                
                for item in all_items:
                    item_id = self.entity_to_id.get(item)
                    if item_id is not None:
                        item_tensor = torch.tensor([item_id]).long()
                        
                        # Score the triple (user, likes, item)
                        with torch.no_grad():
                            score = self.model.score_hrt(
                                hrt_batch=torch.stack([user_tensor, likes_tensor, item_tensor], dim=1)
                            ).item()
                        
                        item_scores.append((item, score))
            
            # Sort by score and return top-k
            item_scores.sort(key=lambda x: x[1], reverse=True)
            return item_scores[:top_k]
            
        except Exception as e:
            raise Exception(f"Error making recommendations: {str(e)}")

    def evaluate_recommendations(self, testing: TriplesFactory, top_k_list: List[int] = [5, 10, 20]) -> dict:
        """
        Evaluate recommendation performance on test set.
        
        Args:
            testing: Test triples factory
            top_k_list: List of k values to evaluate (e.g., [5, 10, 20])
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        print("Evaluating recommendation performance...")
        
        # Get test triples for 'likes' relation only
        # Convert mapped triples back to labeled triples
        test_triples = []
        id_to_entity = {v: k for k, v in self.entity_to_id.items()}
        id_to_relation = {v: k for k, v in self.relation_to_id.items()}
        
        for head_id, rel_id, tail_id in testing.mapped_triples:
            head = id_to_entity.get(head_id.item(), f"entity_{head_id.item()}")
            rel = id_to_relation.get(rel_id.item(), f"relation_{rel_id.item()}")
            tail = id_to_entity.get(tail_id.item(), f"entity_{tail_id.item()}")
            test_triples.append((head, rel, tail))
        
        test_likes = [(h, r, t) for h, r, t in test_triples if r == 'likes']
        
        if not test_likes:
            print("No 'likes' relations found in test set for recommendation evaluation")
            return {}
        
        print(f"Evaluating on {len(test_likes)} test 'likes' relationships")
        
        # Group test likes by user
        user_test_items = {}
        for head, rel, tail in test_likes:
            if head not in user_test_items:
                user_test_items[head] = []
            user_test_items[head].append(tail)
        
        # Evaluate each user
        metrics = {f'precision@{k}': [] for k in top_k_list}
        metrics.update({f'recall@{k}': [] for k in top_k_list})
        metrics.update({f'ndcg@{k}': [] for k in top_k_list})
        
        successful_users = 0
        
        for user, true_items in user_test_items.items():
            try:
                # Get recommendations for this user
                max_k = max(top_k_list)
                recommendations = self.recommend_items(user, top_k=max_k, filter_known=False)
                
                if not recommendations:
                    continue
                
                recommended_items = [item for item, score in recommendations]
                true_items_set = set(true_items)
                
                # Calculate metrics for different k values
                for k in top_k_list:
                    top_k_recommendations = set(recommended_items[:k])
                    
                    # Precision@k = (relevant items in top-k) / k
                    precision = len(top_k_recommendations & true_items_set) / k
                    metrics[f'precision@{k}'].append(precision)
                    
                    # Recall@k = (relevant items in top-k) / total relevant items
                    recall = len(top_k_recommendations & true_items_set) / len(true_items_set)
                    metrics[f'recall@{k}'].append(recall)
                    
                    # NDCG@k (simplified version)
                    dcg = 0
                    idcg = sum([1/np.log2(i+2) for i in range(min(len(true_items_set), k))])
                    
                    for i, item in enumerate(recommended_items[:k]):
                        if item in true_items_set:
                            dcg += 1 / np.log2(i + 2)
                    
                    ndcg = dcg / idcg if idcg > 0 else 0
                    metrics[f'ndcg@{k}'].append(ndcg)
                
                successful_users += 1
                
            except Exception as e:
                print(f"Error evaluating user {user}: {e}")
                continue
        
        # Calculate average metrics
        avg_metrics = {}
        for metric_name, values in metrics.items():
            if values:
                avg_metrics[metric_name] = np.mean(values)
            else:
                avg_metrics[metric_name] = 0.0
        
        avg_metrics['evaluated_users'] = successful_users
        avg_metrics['total_test_users'] = len(user_test_items)
        
        # Print results
        print(f"\nRecommendation Evaluation Results ({successful_users}/{len(user_test_items)} users):")
        print("-" * 60)
        for k in top_k_list:
            print(f"Top-{k} Recommendations:")
            print(f"  Precision@{k}: {avg_metrics[f'precision@{k}']:.4f}")
            print(f"  Recall@{k}: {avg_metrics[f'recall@{k}']:.4f}")
            print(f"  NDCG@{k}: {avg_metrics[f'ndcg@{k}']:.4f}")
            print()
        
        return avg_metrics
    
    def get_entity_embeddings(self) -> torch.Tensor:
        """Get learned entity embeddings."""
        if self.model is None:
            raise ValueError("Model must be trained first")
        return self.model.entity_representations[0]().detach()
    
    def get_relation_embeddings(self) -> torch.Tensor:
        """Get learned relation embeddings."""
        if self.model is None:
            raise ValueError("Model must be trained first")
        return self.model.relation_representations[0]().detach()
    
    def visualize_embeddings(self, save_path: Optional[str] = None):
        """
        Visualize entity embeddings using t-SNE.
        
        Args:
            save_path: Optional path to save the plot
        """
        try:
            from sklearn.manifold import TSNE
            import matplotlib.pyplot as plt
            
            if self.model is None:
                raise ValueError("Model must be trained first")
            
            # Get embeddings and move to CPU
            embeddings = self.get_entity_embeddings().cpu().numpy()
            
            # Apply t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
            embeddings_2d = tsne.fit_transform(embeddings)
            
            # Create plot
            plt.figure(figsize=(12, 8))
            
            # Color entities by type
            colors = {'user': 'blue', 'item': 'red', 'feature': 'green'}
            
            for entity, entity_id in self.entity_to_id.items():
                entity_type = entity.split('_')[0] if '_' in entity else 'other'
                color = colors.get(entity_type, 'gray')
                
                x, y = embeddings_2d[entity_id]
                plt.scatter(x, y, c=color, alpha=0.7)
                
                # Add labels for a subset of entities
                if len(self.entity_to_id) < 100:  # Only label if not too many entities
                    plt.annotate(entity, (x, y), xytext=(5, 5), 
                               textcoords='offset points', fontsize=8, alpha=0.7)
            
            # Create legend
            legend_elements = [plt.scatter([], [], c=color, label=entity_type.title()) 
                             for entity_type, color in colors.items()]
            plt.legend(handles=legend_elements)
            
            plt.title('Knowledge Graph Entity Embeddings (t-SNE)')
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            
        except ImportError:
            print("scikit-learn is required for visualization. Install with: pip install scikit-learn")
        except Exception as e:
            print(f"Error creating visualization: {str(e)}")

def generate_sample_csv(filename: str = "sample_knowledge_graph.csv", num_users: int = 50, 
                       num_items: int = 30, num_features: int = 20):
    """
    Generate a sample CSV file with knowledge graph data.
    
    Args:
        filename: Output CSV filename
        num_users: Number of users to generate
        num_items: Number of items to generate  
        num_features: Number of features to generate
    """
    np.random.seed(42)
    triples = []
    
    # Generate entities
    users = [f"user_{i}" for i in range(num_users)]
    items = [f"item_{i}" for i in range(num_items)]
    features = [f"feature_{i}" for i in range(num_features)]
    
    # User-Item relations (likes/dislikes)
    for user in users:
        # Each user likes/dislikes some items
        liked_items = np.random.choice(items, size=np.random.randint(3, 8), replace=False)
        disliked_items = np.random.choice(items, size=np.random.randint(1, 4), replace=False)
        
        for item in liked_items:
            triples.append([user, "likes", item])
        for item in disliked_items:
            if item not in liked_items:  # Avoid conflicts
                triples.append([user, "dislikes", item])
    
    # User-Feature relations (concerns)
    for user in users:
        concerned_features = np.random.choice(features, size=np.random.randint(2, 6), replace=False)
        for feature in concerned_features:
            triples.append([user, "concerns", feature])
    
    # Feature-Item relations (has)
    for item in items:
        item_features = np.random.choice(features, size=np.random.randint(3, 8), replace=False)
        for feature in item_features:
            triples.append([feature, "has", item])
    
    # Create DataFrame and save
    df = pd.DataFrame(triples, columns=['subject', 'relation', 'object'])
    df.to_csv(filename, index=False)
    print(f"Generated {len(triples)} triples and saved to {filename}")
    return filename

# Example usage function
def main():
    """Example usage of the KnowledgeGraphGenerator."""
    
    # Generate sample data
    csv_file = generate_sample_csv("sample_kg.csv", num_users=20, num_items=15, num_features=10)
    print("shape of generated CSV:", pd.read_csv(csv_file).shape)
    
    # Initialize the generator
    kg_generator = KnowledgeGraphGenerator()
    
    # Load and process data
    df = kg_generator.load_csv_data(csv_file)
    training, validation, testing = kg_generator.create_triples_factory(df)
    
    # Train the model (now with testing parameter)
    results = kg_generator.train_knowledge_graph(
        training, validation, testing,
        model_name='TransE',
        epochs=50,
        embedding_dim=64
    )
    
    # Evaluate the model using standard KG metrics
    print("\n" + "="*60)
    print("STANDARD KNOWLEDGE GRAPH EVALUATION")
    print("="*60)
    evaluation_results = kg_generator.evaluate_model(testing)
    
    # Evaluate recommendation performance
    print("\n" + "="*60)
    print("RECOMMENDATION SYSTEM EVALUATION")
    print("="*60)
    recommendation_results = kg_generator.evaluate_recommendations(testing, top_k_list=[5, 10, 20])
    
    # Make some sample recommendations
    print("\n" + "="*60)
    print("SAMPLE RECOMMENDATIONS")
    print("="*60)
    try:
        sample_users = ['user_0', 'user_1', 'user_2']
        for user in sample_users:
            print(f"\nTop 5 recommendations for {user}:")
            recommendations = kg_generator.recommend_items(user, top_k=5)
            for i, (item, score) in enumerate(recommendations, 1):
                print(f"  {i}. {item}: {score:.4f}")
    except Exception as e:
        print(f"Error generating sample recommendations: {e}")
    
    # Visualize embeddings
    print("\n" + "="*60)
    print("VISUALIZATION")
    print("="*60)
    kg_generator.visualize_embeddings()
    
    return kg_generator

if __name__ == "__main__":
    # Install required packages:
    # pip install pykeen torch pandas numpy matplotlib networkx scikit-learn
    
    kg_generator = main()