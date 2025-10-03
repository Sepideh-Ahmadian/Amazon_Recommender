import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import numpy as np
import warnings
import pickle
from collections import defaultdict
import random
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')

def analyze_kg_data_structure(kg_data):
    """Analyze the structure of knowledge graph data to understand rating distribution"""
    print("=" * 80)
    print("KNOWLEDGE GRAPH STRUCTURE ANALYSIS")
    print("=" * 80)
    
    # Analyze all relations
    relation_counts = defaultdict(int)
    rating_relations = []
    
    for triple in kg_data['triples']:
        head_str, relation_str, tail_str = triple
        relation_counts[relation_str] += 1
        
        # Check for rating patterns
        if 'rate' in relation_str.lower():
            rating_relations.append(relation_str)
    
    print("Top 20 relations by frequency:")
    sorted_relations = sorted(relation_counts.items(), key=lambda x: x[1], reverse=True)
    for rel, count in sorted_relations[:20]:
        print(f"  {rel}: {count}")
    
    print(f"\nRating relations found: {len(set(rating_relations))}")
    rating_dist = defaultdict(int)
    for rel in rating_relations:
        rating_dist[rel] += 1
    
    print("Rating distribution:")
    for rel, count in sorted(rating_dist.items()):
        print(f"  {rel}: {count}")
    
    return relation_counts, rating_relations

def create_improved_recommendation_data(kg_data, min_rating_threshold=3.0, max_rating_threshold=5.0, 
                                      min_user_interactions=5, min_item_interactions=3):
    """
    Create recommendation data with improved filtering and negative sampling
    """
    print("\n" + "=" * 80)
    print("CREATING IMPROVED RECOMMENDATION DATASET")
    print("=" * 80)
    
    # First, analyze the data structure
    relation_counts, rating_relations = analyze_kg_data_structure(kg_data)
    
    # Extract user and item mappings
    user_entities = {k: int(v) for k, v in kg_data.get('user_entities', {}).items()}
    item_entities = {k: int(v) for k, v in kg_data.get('item_entities', {}).items()}
    
    if not user_entities or not item_entities:
        print("❌ No user or item entities found!")
        return None
    
    user_ids = set(user_entities.values())
    item_ids = set(item_entities.values())
    
    print(f"Initial: {len(user_ids)} users, {len(item_ids)} items")
    
    # Extract ALL rating interactions with actual rating values
    all_rating_interactions = []
    
    for triple in kg_data['triples']:
        head_str, relation_str, tail_str = triple
        head_id = kg_data['entity_to_id'][head_str]
        tail_id = kg_data['entity_to_id'][tail_str]
        
        # More flexible rating extraction
        if 'rate' in relation_str.lower():
            # Try to extract numeric rating
            rating_value = None
            
            # Method 1: Extract from relation string
            import re
            numbers = re.findall(r'\d+\.?\d*', relation_str)
            if numbers:
                try:
                    rating_value = float(numbers[0])
                    # Normalize to 1-5 scale if needed
                    if rating_value > 5:
                        rating_value = rating_value / 2  # Assuming 10-point scale
                except ValueError:
                    continue
            
            # Method 2: Check for star patterns
            elif 'star' in relation_str.lower():
                stars = re.findall(r'(\d+)\s*star', relation_str.lower())
                if stars:
                    rating_value = float(stars[0])
            
            if rating_value is not None:
                # Check if this is a user-item interaction
                if head_id in user_ids and tail_id in item_ids:
                    all_rating_interactions.append((head_id, tail_id, rating_value))
                elif tail_id in user_ids and head_id in item_ids:
                    all_rating_interactions.append((tail_id, head_id, rating_value))
    
    print(f"Extracted {len(all_rating_interactions)} rating interactions")
    
    if len(all_rating_interactions) == 0:
        print("❌ No rating interactions found!")
        return None
    
    # Analyze rating distribution
    ratings = [rating for _, _, rating in all_rating_interactions]
    print(f"Rating statistics:")
    print(f"  Min: {min(ratings):.2f}")
    print(f"  Max: {max(ratings):.2f}")
    print(f"  Mean: {np.mean(ratings):.2f}")
    print(f"  Median: {np.median(ratings):.2f}")
    
    # Count interactions per user/item BEFORE filtering
    user_interaction_counts = defaultdict(int)
    item_interaction_counts = defaultdict(int)
    
    for user_id, item_id, rating in all_rating_interactions:
        user_interaction_counts[user_id] += 1
        item_interaction_counts[item_id] += 1
    
    # Filter users and items with sufficient interactions
    valid_users = {u for u, count in user_interaction_counts.items() if count >= min_user_interactions}
    valid_items = {i for i, count in item_interaction_counts.items() if count >= min_item_interactions}
    
    print(f"After filtering (min {min_user_interactions} user interactions, {min_item_interactions} item interactions):")
    print(f"  Valid users: {len(valid_users)}")
    print(f"  Valid items: {len(valid_items)}")
    
    # Filter interactions to only include valid users and items
    filtered_interactions = [
        (u, i, r) for u, i, r in all_rating_interactions 
        if u in valid_users and i in valid_items
    ]
    
    print(f"Filtered interactions: {len(filtered_interactions)}")
    
    if len(filtered_interactions) < 100:
        print("❌ Too few interactions after filtering!")
        return None
    
    # Create positive and negative samples with balanced approach
    positive_interactions = []
    negative_interactions = []
    
    # Determine thresholds based on data distribution
    rating_values = [r for _, _, r in filtered_interactions]
    rating_median = np.median(rating_values)
    rating_mean = np.mean(rating_values)
    
    # Adaptive thresholds
    pos_threshold = max(min_rating_threshold, rating_median)
    neg_threshold = min(max_rating_threshold, rating_mean)
    
    print(f"Using thresholds: positive >= {pos_threshold:.2f}, negative < {neg_threshold:.2f}")
    
    for user_id, item_id, rating in filtered_interactions:
        if rating >= pos_threshold:
            positive_interactions.append((user_id, item_id, 1))
        elif rating < neg_threshold:
            negative_interactions.append((user_id, item_id, 0))
    
    print(f"Natural interactions:")
    print(f"  Positive: {len(positive_interactions)}")
    print(f"  Negative: {len(negative_interactions)}")
    
    # Apply undersampling to balance the dataset
    # Instead of generating more negatives, reduce positives to match negatives
    target_samples = min(len(positive_interactions), len(negative_interactions))
    
    print(f"Applying undersampling:")
    print(f"  Target samples per class: {target_samples}")
    
    # Randomly sample from positive and negative interactions
    if len(positive_interactions) > target_samples:
        positive_interactions = random.sample(positive_interactions, target_samples)
        print(f"  Reduced positive samples from {len(positive_interactions)} to {target_samples}")
    
    if len(negative_interactions) > target_samples:
        negative_interactions = random.sample(negative_interactions, target_samples)
        print(f"  Reduced negative samples from {len(negative_interactions)} to {target_samples}")
    
    # Combine all interactions (no additional negatives generated)
    all_interactions = positive_interactions + negative_interactions
    random.shuffle(all_interactions)
    
    print(f"\nFinal dataset:")
    print(f"  Total interactions: {len(all_interactions)}")
    print(f"  Positive: {len(positive_interactions)} ({len(positive_interactions)/len(all_interactions)*100:.1f}%)")
    print(f"  Negative: {len(negative_interactions)} ({len(negative_interactions)/len(all_interactions)*100:.1f}%)")
    
    # Create mappings for the filtered data
    user_to_new_id = {user: i for i, user in enumerate(sorted(valid_users))}
    item_to_new_id = {item: i + len(user_to_new_id) for i, item in enumerate(sorted(valid_items))}
    
    # Create PyTorch tensors with remapped IDs
    user_indices = torch.tensor([user_to_new_id[u] for u, i, l in all_interactions], dtype=torch.long)
    item_indices = torch.tensor([item_to_new_id[i] for u, i, l in all_interactions], dtype=torch.long)
    labels = torch.tensor([l for u, i, l in all_interactions], dtype=torch.float)
    
    # Create node features for both users and items
    num_entities = len(user_to_new_id) + len(item_to_new_id)
    x = torch.arange(num_entities, dtype=torch.long)
    
    # Create edge index from KG (only for entities that remain)
    edge_index = create_filtered_edge_index(kg_data, valid_users | valid_items, user_to_new_id, item_to_new_id)
    
    graph_data = Data(
        x=x,
        edge_index=edge_index,
        user_indices=user_indices,
        item_indices=item_indices,
        labels=labels,
        num_users=len(valid_users),
        num_items=len(valid_items),
        user_to_original=user_to_new_id,
        item_to_original=item_to_new_id
    )
    
    return graph_data

def create_filtered_edge_index(kg_data, valid_entities, user_mapping, item_mapping):
    """Create edge index only for valid entities"""
    source_nodes = []
    target_nodes = []
    
    # Combined mapping
    entity_mapping = {**user_mapping, **item_mapping}
    
    for head_str, relation_str, tail_str in kg_data['triples']:
        if head_str in kg_data['entity_to_id'] and tail_str in kg_data['entity_to_id']:
            head_id = kg_data['entity_to_id'][head_str]
            tail_id = kg_data['entity_to_id'][tail_str]
            
            # Only include edges between valid entities
            if head_id in valid_entities and tail_id in valid_entities:
                if head_id in entity_mapping and tail_id in entity_mapping:
                    source_nodes.append(entity_mapping[head_id])
                    target_nodes.append(entity_mapping[tail_id])
    
    if len(source_nodes) == 0:
        # Create a minimal self-loop structure
        num_entities = len(entity_mapping)
        source_nodes = list(range(num_entities))
        target_nodes = list(range(num_entities))
    
    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    return edge_index

def create_user_based_data_splits(graph_data, train_ratio=0.7, val_ratio=0.15):
    """
    Create data splits ensuring each user appears in only one split
    """
    # Get unique users
    unique_users = torch.unique(graph_data.user_indices)
    num_users = len(unique_users)
    
    # Shuffle users
    shuffled_indices = torch.randperm(num_users)
    shuffled_users = unique_users[shuffled_indices]
    
    # Split users
    train_user_count = int(train_ratio * num_users)
    val_user_count = int(val_ratio * num_users)
    
    train_users = set(shuffled_users[:train_user_count].tolist())
    val_users = set(shuffled_users[train_user_count:train_user_count + val_user_count].tolist())
    test_users = set(shuffled_users[train_user_count + val_user_count:].tolist())
    
    # Create masks for interactions
    train_mask = torch.tensor([user.item() in train_users for user in graph_data.user_indices])
    val_mask = torch.tensor([user.item() in val_users for user in graph_data.user_indices])
    test_mask = torch.tensor([user.item() in test_users for user in graph_data.user_indices])
    
    # Create data splits
    def create_split_data(mask):
        return Data(
            x=graph_data.x,
            edge_index=graph_data.edge_index,
            user_indices=graph_data.user_indices[mask],
            item_indices=graph_data.item_indices[mask],
            labels=graph_data.labels[mask]
        )
    
    train_data = create_split_data(train_mask)
    val_data = create_split_data(val_mask)
    test_data = create_split_data(test_mask)
    
    print(f"\nUser-based data splits:")
    print(f"  Train: {len(train_users)} users, {train_mask.sum().item()} interactions")
    print(f"  Val: {len(val_users)} users, {val_mask.sum().item()} interactions")
    print(f"  Test: {len(test_users)} users, {test_mask.sum().item()} interactions")
    
    return train_data, val_data, test_data

class RecommendationEvaluator:
    """Evaluation metrics for recommendation systems"""
    
    @staticmethod
    def precision_at_k(predictions, targets, k):
        """Calculate Precision@K"""
        k = min(k, len(predictions))
        _, top_k_indices = torch.topk(predictions, k)
        top_k_targets = targets[top_k_indices]
        return (top_k_targets.sum().float() / k).item()
    
    @staticmethod
    def recall_at_k(predictions, targets, k):
        """Calculate Recall@K"""
        k = min(k, len(predictions))
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
        sorted_targets = targets[sorted_indices[:k]]
        
        relevant_items = torch.cumsum(sorted_targets, dim=0)
        precision_at_i = relevant_items.float() / torch.arange(1, k + 1, device=predictions.device).float()
        
        total_relevant = targets.sum().float()
        if total_relevant == 0:
            return 0.0
            
        map_score = (precision_at_i * sorted_targets).sum() / total_relevant
        return map_score.item() if not torch.isnan(map_score) else 0.0
    
    @staticmethod
    def ndcg_at_k(predictions, targets, k):
        """Calculate NDCG@K (Normalized Discounted Cumulative Gain)"""
        k = min(k, len(predictions))
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
        
        if idcg == 0:
            return 0.0
        
        ndcg = (dcg / idcg).item()
        return ndcg if not torch.isnan(torch.tensor(ndcg)) else 0.0

# Simplified GAT model for better performance
class SimpleAdvancedGAT(nn.Module):
    """Simplified but effective GAT model"""
    
    def __init__(self, num_nodes, embedding_dim=64, hidden_dim=128, num_heads=4, num_layers=2, dropout=0.3):
        super(SimpleAdvancedGAT, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_nodes = num_nodes
        
        # Node embeddings
        self.node_embedding = nn.Embedding(num_nodes, embedding_dim)
        nn.init.xavier_uniform_(self.node_embedding.weight)
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        
        # First layer
        self.gat_layers.append(
            GATConv(embedding_dim, hidden_dim // num_heads, heads=num_heads, 
                   dropout=dropout, concat=True)
        )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.gat_layers.append(
                GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, 
                       dropout=dropout, concat=True)
            )
        
        # Output layer
        if num_layers > 1:
            self.gat_layers.append(
                GATConv(hidden_dim, embedding_dim, heads=1, dropout=dropout, concat=False)
            )
        
        # Prediction layers
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout // 2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, user_indices, item_indices):
        # Get node embeddings
        h = self.node_embedding(x)
        
        # Apply GAT layers
        for i, gat_layer in enumerate(self.gat_layers):
            h = gat_layer(h, edge_index)
            if i < len(self.gat_layers) - 1:
                h = F.elu(h)
                h = self.dropout(h)
        
        # Get user and item embeddings
        user_emb = h[user_indices]
        item_emb = h[item_indices]
        
        # Combine and predict
        combined = torch.cat([user_emb, item_emb], dim=1)
        predictions = self.predictor(combined)
        
        return predictions.squeeze()

class GATTrainer:
    """Enhanced trainer with user-based evaluation and @K metrics"""
    
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
        logits = self.model(x, edge_index, user_indices, item_indices)
        
        # Calculate loss
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()
        
        return loss.item()
    
    def evaluate_per_user(self, data, k_values=[5, 10, 20]):
        """Evaluate the model per user and average the results"""
        self.model.eval()
        
        with torch.no_grad():
            x = data.x.to(self.device)
            edge_index = data.edge_index.to(self.device)
            
            # Group interactions by user
            user_groups = defaultdict(list)
            for i, (user_idx, item_idx, label) in enumerate(zip(data.user_indices, data.item_indices, data.labels)):
                user_groups[user_idx.item()].append((item_idx.item(), label.item(), i))
            
            metrics_per_k = {f'Precision@{k}': [] for k in k_values}
            metrics_per_k.update({f'Recall@{k}': [] for k in k_values})
            metrics_per_k.update({f'MAP@{k}': [] for k in k_values})
            metrics_per_k.update({f'NDCG@{k}': [] for k in k_values})
            
            for user_idx, interactions in user_groups.items():
                if len(interactions) < max(k_values):  # Skip users with too few interactions
                    continue
                
                # Extract items and labels for this user
                item_indices = torch.tensor([item_idx for item_idx, _, _ in interactions], device=self.device)
                labels = torch.tensor([label for _, label, _ in interactions], device=self.device)
                user_indices = torch.full((len(interactions),), user_idx, device=self.device)
                
                # Get predictions for this user
                logits = self.model(x, edge_index, user_indices, item_indices)
                predictions = torch.sigmoid(logits)
                
                # Calculate metrics for each k
                for k in k_values:
                    if len(predictions) >= k:
                        metrics_per_k[f'Precision@{k}'].append(
                            self.evaluator.precision_at_k(predictions, labels, k)
                        )
                        metrics_per_k[f'Recall@{k}'].append(
                            self.evaluator.recall_at_k(predictions, labels, k)
                        )
                        metrics_per_k[f'MAP@{k}'].append(
                            self.evaluator.map_at_k(predictions, labels, k)
                        )
                        metrics_per_k[f'NDCG@{k}'].append(
                            self.evaluator.ndcg_at_k(predictions, labels, k)
                        )
            
            # Average metrics across users
            final_metrics = {}
            for metric_name, values in metrics_per_k.items():
                if values:  # Only calculate if we have values
                    final_metrics[metric_name] = np.mean(values)
                else:
                    final_metrics[metric_name] = 0.0
            
            return final_metrics
    
    def train_and_evaluate(self, train_data, val_data, test_data, epochs=200, lr=0.001, k_values=[5, 10, 20]):
        """Complete training and evaluation pipeline"""
        
        # Calculate class weights
        pos_count = (train_data.labels == 1).sum().float()
        neg_count = (train_data.labels == 0).sum().float()
        pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        
        print(f"Class weights: pos_weight = {pos_weight:.3f}")
        
        # Setup optimizer and loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        pos_weight_tensor = torch.tensor([pos_weight], device=self.device)
        
        def criterion(logits, labels):
            return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight_tensor)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        best_val_ndcg = 0
        best_metrics = None
        patience_counter = 0
        patience = 20
        
        print("Starting training...")
        print("-" * 60)
        
        for epoch in range(epochs):
            # Training
            train_loss = self.train_epoch(train_data, optimizer, criterion)
            
            # Validation every 10 epochs
            if epoch % 10 == 0:
                val_metrics = self.evaluate_per_user(val_data, k_values)
                
                available_k = k_values[0]  # Use first k value
                val_ndcg = val_metrics.get(f'NDCG@{available_k}', 0.0)
                
                print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val NDCG@{available_k}: {val_ndcg:.4f}")
                
                scheduler.step(val_ndcg)
                
                # Save best model
                if val_ndcg > best_val_ndcg:
                    best_val_ndcg = val_ndcg
                    best_metrics = val_metrics.copy()
                    patience_counter = 0
                    torch.save(self.model.state_dict(), 'best_enhanced_model.pth')
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print("Early stopping!")
                    break
        
        # Load best model for final evaluation
        self.model.load_state_dict(torch.load('best_enhanced_model.pth'))
        
        # Final evaluation on test set
        print("\nFinal Evaluation on Test Set:")
        print("=" * 60)
        
        test_metrics = self.evaluate_per_user(test_data, k_values)
        
        for metric_name, value in test_metrics.items():
            print(f"{metric_name:12s}: {value:.4f}")
        
        return test_metrics

def run_enhanced_recommendation_system(embeddings_path: str):
    """Run the enhanced recommendation system with user splitting and @K metrics"""
    
    print("=" * 100)
    print("ENHANCED GAT RECOMMENDATION SYSTEM")
    print("=" * 100)
    
    try:
        # Load data
        with open(embeddings_path, 'rb') as f:
            kg_data = pickle.load(f)
        
        # Create improved dataset
        graph_data = create_improved_recommendation_data(
            kg_data,
            min_rating_threshold=3.0,
            max_rating_threshold=5.0,
            min_user_interactions=1,
            min_item_interactions=1
        )
        
        if graph_data is None:
            print("❌ Failed to create improved dataset")
            return None
        
        # Create user-based data splits
        train_data, val_data, test_data = create_user_based_data_splits(graph_data)
        
        # Initialize improved model
        model = SimpleAdvancedGAT(
            num_nodes=len(graph_data.x),
            embedding_dim=64,
            hidden_dim=128,
            num_heads=4,
            num_layers=2,
            dropout=0.3
        )
        
        print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Initialize enhanced trainer
        trainer = GATTrainer(model)
        
        # Train and evaluate with @K metrics
        final_metrics = trainer.train_and_evaluate(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            epochs=200,
            lr=0.001,
            k_values=[5, 10, 20]
        )
        
        print("\n✨ Enhanced training completed!")
        print("\n📊 Final Test Metrics:")
        print("-" * 40)
        for metric_name, value in final_metrics.items():
            print(f"{metric_name}: {value:.4f}")
        
        return model, final_metrics
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    embeddings_path = 'Data/Knowledge graph representations/KG_embeddings_for_gnn.pkl'
    result = run_enhanced_recommendation_system(embeddings_path)
    
    if result:
        model, metrics = result
        print("\n💫 Enhanced pipeline completed successfully!")
        print("You can now use the trained model for recommendations with proper evaluation metrics.")
        #undersampling the results are also reasonable
