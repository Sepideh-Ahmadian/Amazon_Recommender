import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import numpy as np
from sklearn.metrics import precision_recall_curve
import warnings
import pickle

warnings.filterwarnings('ignore')

class GATRecommendationModel(nn.Module):
    """Graph Attention Network for Recommendation"""
    
    def __init__(self, num_nodes, embedding_dim=64, hidden_dim=128, num_heads=4, num_layers=2, dropout=0.1):
        super(GATRecommendationModel, self).__init__()
        
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Node embeddings
        self.node_embedding = nn.Embedding(num_nodes, embedding_dim)
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        
        # First layer
        self.gat_layers.append(
            GATConv(embedding_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout, concat=True)
        )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.gat_layers.append(
                GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout, concat=True)
            )
        
        # Output layer
        if num_layers > 1:
            self.gat_layers.append(
                GATConv(hidden_dim, embedding_dim, heads=1, dropout=dropout, concat=False)
            )
        
        # Prediction layer
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, user_indices, item_indices):
        # Get initial node embeddings
        h = self.node_embedding(x)
        
        # Apply GAT layers
        for i, gat_layer in enumerate(self.gat_layers):
            h = gat_layer(h, edge_index)
            if i < len(self.gat_layers) - 1:  # Don't apply activation to last layer
                h = F.elu(h)
                h = self.dropout(h)
        
        # Get user and item embeddings
        user_embeddings = h[user_indices]
        item_embeddings = h[item_indices]
        
        # Concatenate user and item embeddings
        combined = torch.cat([user_embeddings, item_embeddings], dim=1)
        
        # Predict interaction probability
        predictions = self.predictor(combined)
        
        return predictions.squeeze()

class RecommendationEvaluator:
    """Evaluation metrics for recommendation systems"""
    
    @staticmethod
    def precision_at_k(predictions, targets, k):
        k = min(k, len(predictions))
        """Calculate Precision@K"""
        # Get top-k predictions
        _, top_k_indices = torch.topk(predictions, k)
        top_k_targets = targets[top_k_indices]
        return (top_k_targets.sum().float() / k).item()
    
    @staticmethod
    def recall_at_k(predictions, targets, k):
        k = min(k, len(predictions))
        """Calculate Recall@K"""
        if targets.sum() == 0:
            return 0.0
        _, top_k_indices = torch.topk(predictions, k)
        top_k_targets = targets[top_k_indices]
        return (top_k_targets.sum().float() / targets.sum().float()).item()
    
    @staticmethod
    def map_at_k(predictions, targets, k):
        """Calculate MAP@K (Mean Average Precision)"""
        k = min(k, len(predictions))
        _, sorted_indices = torch.sort(predictions, descending=True)
        sorted_targets = targets[sorted_indices]
        
        relevant_items = torch.cumsum(sorted_targets, dim=0)
        precision_at_i = relevant_items.float() / torch.arange(1, len(sorted_targets) + 1).float()
        
        # Only consider relevant items for MAP calculation
        map_score = (precision_at_i * sorted_targets).sum() / min(targets.sum().float(), k)
        return map_score.item() if not torch.isnan(map_score) else 0.0
    
    @staticmethod
    def ndcg_at_k(predictions, targets, k):
        """Calculate NDCG@K (Normalized Discounted Cumulative Gain)"""
        # Get top-k predictions and ensure k doesn't exceed predictions length
        k = min(k, len(predictions))
        print(k)
        _, top_k_indices = torch.topk(predictions, k)
        top_k_targets = targets[top_k_indices]
        
        # Calculate DCG
        gains = top_k_targets.float()
        discounts = torch.log2(torch.arange(2, k + 2, device=predictions.device).float())
        dcg = (gains / discounts).sum()
        
        # Calculate IDCG (Ideal DCG)
        ideal_gains, _ = torch.sort(targets.float(), descending=True)
        ideal_gains = ideal_gains[:k]
        idcg = (ideal_gains / discounts[:len(ideal_gains)]).sum()
        
        # Handle zero division case
        if idcg == 0:
            return 0.0
        
        ndcg = (dcg / idcg).item()
        print(ndcg)
        return ndcg if not torch.isnan(torch.tensor(ndcg)) else 0.0

class GATRecommendationTrainer:
    """Training and evaluation pipeline for GAT recommendation model"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.evaluator = RecommendationEvaluator()
        
    def train_epoch(self, data, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        user_indices = data.user_indices.to(self.device)
        item_indices = data.item_indices.to(self.device)
        labels = data.labels.to(self.device).float()
        
        optimizer.zero_grad()
        
        # Forward pass
        predictions = self.model(x, edge_index, user_indices, item_indices)
        
        # Calculate loss
        loss = criterion(predictions, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def evaluate(self, data, k_values=[5, 10, 20]):
        """Evaluate the model with multiple metrics"""
        self.model.eval()
        
        with torch.no_grad():
            x = data.x.to(self.device)
            edge_index = data.edge_index.to(self.device)
            user_indices = data.user_indices.to(self.device)
            item_indices = data.item_indices.to(self.device)
            labels = data.labels.to(self.device)
            
            # Get predictions
            predictions = self.model(x, edge_index, user_indices, item_indices)
            
            # Calculate metrics for each k
            metrics = {}
            for k in k_values:
                k = min(k, len(predictions))  # Ensure k doesn't exceed number of items
                
                metrics[f'Precision@{k}'] = self.evaluator.precision_at_k(predictions, labels, k)
                metrics[f'Recall@{k}'] = self.evaluator.recall_at_k(predictions, labels, k)
                metrics[f'MAP@{k}'] = self.evaluator.map_at_k(predictions, labels, k)
                metrics[f'NDCG@{k}'] = self.evaluator.ndcg_at_k(predictions, labels, k)
            
            return metrics
    
    def train_and_evaluate(self, train_data, val_data, test_data, epochs=100, lr=0.001, k_values=[5, 10, 20]):
        """Complete training and evaluation pipeline"""
        
        # Setup optimizer and loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.BCELoss()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
        
        best_val_ndcg = 0
        best_metrics = None
        
        print("Starting training...")
        print("-" * 60)
        
        for epoch in range(epochs):
            # Training
            train_loss = self.train_epoch(train_data, optimizer, criterion)
            
            # Validation every 10 epochs
            if epoch % 10 == 0:
                val_metrics = self.evaluate(val_data, k_values)
                
                # Use the smallest available k value that exists in metrics
                available_k = None
                for k in k_values:
                    if f'NDCG@{k}' in val_metrics:
                        available_k = k
                        break
                
                if available_k is not None:
                    val_ndcg = val_metrics[f'NDCG@{available_k}']
                    print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val NDCG@{available_k}: {val_ndcg:.4f}")
                    
                    # Save best model
                    if val_ndcg > best_val_ndcg:
                        best_val_ndcg = val_ndcg
                        best_metrics = val_metrics.copy()
                        # Here you would typically save the model state
                        # torch.save(self.model.state_dict(), 'best_model.pt')
                else:
                    print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val NDCG: N/A (insufficient data)")
            
            # Learning rate scheduling
            scheduler.step()
        
        # Final evaluation on test set
        print("\nFinal Evaluation on Test Set:")
        print("=" * 60)
        
        test_metrics = self.evaluate(test_data, k_values)
        
        for metric_name, value in test_metrics.items():
            print(f"{metric_name:12s}: {value:.4f}")
        
        return test_metrics
        

def convert_triples_to_edge_index(triples, entity_to_id):
                """
                Convert knowledge graph triples to PyTorch Geometric edge_index format.
                
                Args:
                    triples: List of (head, relation, tail) tuples
                    entity_to_id: Dictionary mapping entity names to IDs
                    
                Returns:
                    torch.Tensor: Edge index of shape [2, num_edges]
                """
                source_nodes = []
                target_nodes = []
                
                for head, relation, tail in triples:
                    if head in entity_to_id and tail in entity_to_id:
                        source_nodes.append(entity_to_id[head])
                        target_nodes.append(entity_to_id[tail])
                
                edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
                return edge_index
def KG_main(embeddings_path: str = 'Data/Knowledge graph representations/KG_embeddings_for_gnn.pkl'):
    """
    Main function to run the GAT recommendation pipeline
    
    Args:
        embeddings_path: Path to the saved knowledge graph embeddings
    """
    # Load knowledge graph embeddings
    print("\n📂 Loading knowledge graph embeddings...")
    try:
        with open(embeddings_path, 'rb') as f:
            kg_data = pickle.load(f)
       
        # Convert embeddings to float tensor
        entity_embeddings = torch.FloatTensor(kg_data['entity_embeddings'])
        

            # Usage in your code:
        edge_index = convert_triples_to_edge_index(
                triples=kg_data['triples'],
                entity_to_id=kg_data['entity_to_id']
            )
        # Convert triples to long tensor and ensure they're numerical
        # edge_index = torch.LongTensor(np.array(kg_data['triples'], dtype=np.int64)).t()

        # Get dimensions
        num_entities = len(kg_data['entity_to_id'])
        embedding_dim = int(kg_data['embedding_dim'])  # Ensure integer
        
        # Convert entity mappings
        user_entities = {k: int(v) for k, v in kg_data.get('user_entities', {}).items()}
        item_entities = {k: int(v) for k, v in kg_data.get('item_entities', {}).items()}
        
        print(f"\nDataset statistics:")
        print(f"  Total entities: {num_entities}")
        print(f"  Users: {len(user_entities)}")
        print(f"  Items: {len(item_entities)}")
        print(f"  Embedding dimension: {embedding_dim}")
        print(f"  Number of edges: {edge_index.shape[1]}")
        
        # Create graph data object with required attributes
        graph_data = Data(
            x=torch.arange(num_entities),
            edge_index=edge_index,
            entity_embeddings=entity_embeddings,
            # Add interaction data
            user_indices=torch.tensor([user_entities[k] for k in user_entities.keys()]),
            item_indices=torch.tensor([item_entities[k] for k in item_entities.keys()]),
            # Initialize labels as ones (assuming all interactions are positive)
            labels=torch.ones(len(user_entities))
        )
        
        # Initialize model
        print("\n🔧 Initializing GAT model...")
        model = GATRecommendationModel(
            num_nodes=num_entities,
            embedding_dim=embedding_dim,
            hidden_dim=128,
            num_heads=4,
            num_layers=2,
            dropout=0.1
        )
        
        print(f"Model architecture:")
        
        print(f"  Parameters: {sum(p.numel() for p in model.parameters())}")
        
        # Create train/val/test splits
        print("\n📊 Preparing data splits...")
        indices = torch.randperm(len(user_entities))
        train_size = int(0.7 * len(indices))
        val_size = int(0.15 * len(indices))
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        def create_data_split(graph_data, split_indices):
            return Data(
                x=graph_data.x,
                edge_index=graph_data.edge_index,
                user_indices=graph_data.user_indices[split_indices],
                item_indices=graph_data.item_indices[split_indices],
                labels=graph_data.labels[split_indices]
            )
        # Create data splits
        train_data = create_data_split(graph_data, train_indices)
        val_data = create_data_split(graph_data, val_indices)
        test_data = create_data_split(graph_data, test_indices)
        
        print(f"Data splits:")
        print(f"  Train: {len(train_indices)} samples")
        print(f"  Validation: {len(val_indices)} samples")
        print(f"  Test: {len(test_indices)} samples")
        
        # Initialize trainer
        print("\n🚀 Initializing trainer...")
        trainer = GATRecommendationTrainer(model)
        
        # Train and evaluate
        print("\n⏳ Starting training pipeline...")
        final_metrics = trainer.train_and_evaluate(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            epochs=50,
            lr=0.001,
            k_values=[5, 10, 20]
        )
        
        print("\n✨ Training completed!")
        print("\n📊 Final Test Metrics:")
        print("-" * 40)
        for metric_name, value in final_metrics.items():
            print(f"{metric_name}: {value:.4f}")
        
        return model, final_metrics
        
    except FileNotFoundError:
        print(f"❌ Error: Embeddings file not found at {embeddings_path}")
        return None, None
    # except Exception as e:
    #     print(f"❌ Error during pipeline execution: {str(e)}")
    #     return None, None

if __name__ == "__main__":
    embeddings_path = 'Data/Knowledge graph representations/KG_embeddings_for_gnn.pkl'
    model, metrics = KG_main(embeddings_path)
    
    if metrics:
        print("\n💫 Pipeline completed successfully!")
        print("You can now use the trained model for recommendations.")