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

class CrossDomainRecommendationData:
    """
    Prepare cross-domain recommendation data where:
    - All Beauty: Target domain for recommendations
    - Beauty & Personal Care: Source domain for enrichment (no recommendations from here)
    """
    
    def __init__(self, kg_data):
        self.kg_data = kg_data
        self.all_beauty_users = {}
        self.all_beauty_items = {}
        self.beauty_pc_users = {}
        self.beauty_pc_items = {}
        self.all_beauty_features = {}
        self.beauty_pc_features = {}
        
    def extract_domain_entities(self):
        """Extract and separate entities by domain"""
        print("Extracting domain-specific entities...")
        
        entity_categorization = self.kg_data.get('entity_categorization', {})
        
        if entity_categorization:
            # Use existing categorization
            self.all_beauty_users = entity_categorization.get('all_beauty', {}).get('users', {})
            self.all_beauty_items = entity_categorization.get('all_beauty', {}).get('items', {})
            self.all_beauty_features = entity_categorization.get('all_beauty', {}).get('features', {})
            
            self.beauty_pc_users = entity_categorization.get('beauty_pc', {}).get('users', {})
            self.beauty_pc_items = entity_categorization.get('beauty_pc', {}).get('items', {})
            self.beauty_pc_features = entity_categorization.get('beauty_pc', {}).get('features', {})
        else:
            # Manual extraction if categorization not available
            for entity, entity_id in self.kg_data['entity_to_id'].items():
                if entity.startswith('all_beauty_'):
                    entity_type = entity.replace('all_beauty_', '').split('_')[0]
                    if entity_type == 'user':
                        self.all_beauty_users[entity] = entity_id
                    elif entity_type == 'item':
                        self.all_beauty_items[entity] = entity_id
                    elif entity_type == 'feature':
                        self.all_beauty_features[entity] = entity_id
                elif entity.startswith('beauty_pc_'):
                    entity_type = entity.replace('beauty_pc_', '').split('_')[0]
                    if entity_type == 'user':
                        self.beauty_pc_users[entity] = entity_id
                    elif entity_type == 'item':
                        self.beauty_pc_items[entity] = entity_id
                    elif entity_type == 'feature':
                        self.beauty_pc_features[entity] = entity_id
        
        print(f"All Beauty - Users: {len(self.all_beauty_users)}, Items: {len(self.all_beauty_items)}, Features: {len(self.all_beauty_features)}")
        print(f"Beauty & PC - Users: {len(self.beauty_pc_users)}, Items: {len(self.beauty_pc_items)}, Features: {len(self.beauty_pc_features)}")
        
    def create_cross_domain_interactions(self, min_rating_threshold=3.0, min_user_interactions=3):
        """
        Create training data focusing on All Beauty domain recommendations
        enriched by Beauty & Personal Care knowledge
        """
        print("\nCreating cross-domain recommendation dataset...")
        
        # Extract rating interactions for All Beauty domain only
        all_beauty_interactions = []
        cross_domain_user_mapping = {}  # Map beauty_pc users to all_beauty users if they exist
        
        # First, identify users that exist in both domains
        all_beauty_user_ids = set()
        beauty_pc_user_ids = set()
        
        for entity in self.all_beauty_users.keys():
            # Extract original user ID
            user_id = entity.replace('all_beauty_user_', '')
            all_beauty_user_ids.add(user_id)
            
        for entity in self.beauty_pc_users.keys():
            # Extract original user ID
            user_id = entity.replace('beauty_pc_user_', '')
            beauty_pc_user_ids.add(user_id)
            cross_domain_user_mapping[entity] = f"all_beauty_user_{user_id}"
        
        common_users = all_beauty_user_ids.intersection(beauty_pc_user_ids)
        print(f"Users existing in both domains: {len(common_users)}")
        
        # Extract All Beauty rating interactions
        # Handle different possible keys for triples data
        triples_data = self.kg_data.get('combined_triples', self.kg_data.get('triples', []))
        for triple in triples_data:
            head_str, relation_str, tail_str = triple
            
            if 'rate' in relation_str.lower():
                head_id = self.kg_data['entity_to_id'][head_str]
                tail_id = self.kg_data['entity_to_id'][tail_str]
                
                # Extract rating value
                import re
                numbers = re.findall(r'\d+\.?\d*', relation_str)
                if numbers:
                    try:
                        rating_value = float(numbers[0])
                        if rating_value > 5:
                            rating_value = rating_value / 2
                    except ValueError:
                        continue
                else:
                    continue
                
                # Only consider All Beauty interactions for recommendations
                if (head_str in self.all_beauty_users and tail_str in self.all_beauty_items) or \
                   (tail_str in self.all_beauty_users and head_str in self.all_beauty_items):
                    
                    if head_str in self.all_beauty_users:
                        user_entity, item_entity = head_str, tail_str
                    else:
                        user_entity, item_entity = tail_str, head_str
                    
                    user_id = self.kg_data['entity_to_id'][user_entity]
                    item_id = self.kg_data['entity_to_id'][item_entity]
                    
                    all_beauty_interactions.append((user_id, item_id, rating_value))
        
        print(f"All Beauty rating interactions: {len(all_beauty_interactions)}")
        
        if len(all_beauty_interactions) == 0:
            print("No All Beauty rating interactions found!")
            return None
        
        # Filter by user interaction count
        user_interaction_counts = defaultdict(int)
        for user_id, item_id, rating in all_beauty_interactions:
            user_interaction_counts[user_id] += 1
        
        valid_users = {u for u, count in user_interaction_counts.items() if count >= min_user_interactions}
        filtered_interactions = [(u, i, r) for u, i, r in all_beauty_interactions if u in valid_users]
        
        print(f"After filtering (min {min_user_interactions} interactions): {len(filtered_interactions)} interactions")
        print(f"Valid users: {len(valid_users)}")
        
        # Create positive and negative samples
        positive_interactions = []
        negative_interactions = []
        
        rating_values = [r for _, _, r in filtered_interactions]
        pos_threshold = max(min_rating_threshold, np.median(rating_values))
        
        for user_id, item_id, rating in filtered_interactions:
            if rating >= pos_threshold:
                positive_interactions.append((user_id, item_id, 1))
            else:
                negative_interactions.append((user_id, item_id, 0))
        
        # Balance the dataset
        target_samples = min(len(positive_interactions), len(negative_interactions))
        if len(positive_interactions) > target_samples:
            positive_interactions = random.sample(positive_interactions, target_samples)
        if len(negative_interactions) > target_samples:
            negative_interactions = random.sample(negative_interactions, target_samples)
        
        all_interactions = positive_interactions + negative_interactions
        random.shuffle(all_interactions)
        
        print(f"Final dataset: {len(all_interactions)} interactions")
        print(f"Positive: {len(positive_interactions)}, Negative: {len(negative_interactions)}")
        
        return all_interactions, valid_users
    
    def create_cross_domain_graph(self, interactions, valid_users):
        """
        Create graph data that includes both domains but only recommends All Beauty items
        """
        print("Creating cross-domain graph...")
        
        # Create entity mappings - include ALL entities from both domains
        all_entities = list(self.kg_data['entity_to_id'].keys())
        entity_to_new_id = {entity: i for i, entity in enumerate(all_entities)}
        
        # Create PyTorch tensors for All Beauty interactions only
        user_indices = []
        item_indices = []
        labels = []
        
        for user_id, item_id, label in interactions:
            # Find entity names for these IDs
            user_entity = None
            item_entity = None
            
            for entity, eid in self.kg_data['entity_to_id'].items():
                if eid == user_id and entity in self.all_beauty_users:
                    user_entity = entity
                elif eid == item_id and entity in self.all_beauty_items:
                    item_entity = entity
            
            if user_entity and item_entity:
                user_indices.append(entity_to_new_id[user_entity])
                item_indices.append(entity_to_new_id[item_entity])
                labels.append(label)
        
        # Create edge index using ALL triplets (both domains)
        source_nodes = []
        target_nodes = []
        
        # Handle different possible keys for triples data
        triples_data = self.kg_data.get('combined_triples', self.kg_data.get('triples', []))
        for head_str, relation_str, tail_str in triples_data:
            if head_str in entity_to_new_id and tail_str in entity_to_new_id:
                source_nodes.append(entity_to_new_id[head_str])
                target_nodes.append(entity_to_new_id[tail_str])
        
        # Create node features
        num_entities = len(all_entities)
        x = torch.arange(num_entities, dtype=torch.long)
        
        # Convert to tensors
        user_indices = torch.tensor(user_indices, dtype=torch.long)
        item_indices = torch.tensor(item_indices, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.float)
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
        
        graph_data = Data(
            x=x,
            edge_index=edge_index,
            user_indices=user_indices,
            item_indices=item_indices,
            labels=labels,
            num_all_beauty_users=len(self.all_beauty_users),
            num_all_beauty_items=len(self.all_beauty_items),
            num_beauty_pc_users=len(self.beauty_pc_users),
            num_beauty_pc_items=len(self.beauty_pc_items),
            entity_to_new_id=entity_to_new_id,
            all_beauty_items=self.all_beauty_items,
            all_beauty_users=self.all_beauty_users
        )
        
        return graph_data

def create_cross_domain_recommendation_data(kg_data, min_rating_threshold=3.0, min_user_interactions=3):
    """
    Main function to create cross-domain recommendation data
    """
    print("=" * 80)
    print("CROSS-DOMAIN RECOMMENDATION DATA CREATION")
    print("=" * 80)
    
    # Initialize cross-domain data processor
    cross_domain_data = CrossDomainRecommendationData(kg_data)
    
    # Extract domain entities
    cross_domain_data.extract_domain_entities()
    
    # Create cross-domain interactions (only All Beauty for recommendations)
    interactions, valid_users = cross_domain_data.create_cross_domain_interactions(
        min_rating_threshold=min_rating_threshold,
        min_user_interactions=min_user_interactions
    )
    
    if interactions is None:
        return None
    
    # Create cross-domain graph
    graph_data = cross_domain_data.create_cross_domain_graph(interactions, valid_users)
    
    return graph_data

class CrossDomainGAT(nn.Module):
    """
    GAT model designed for cross-domain recommendations
    Uses knowledge from both domains but only recommends All Beauty items
    """
    
    def __init__(self, num_nodes, embedding_dim=64, hidden_dim=128, num_heads=4, num_layers=2, dropout=0.3):
        super(CrossDomainGAT, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_nodes = num_nodes
        
        # Node embeddings - includes ALL entities from both domains
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
        
        # Cross-domain fusion layer
        self.cross_domain_fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim)
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
        # Get node embeddings for ALL entities (both domains)
        h = self.node_embedding(x)
        
        # Apply GAT layers - this processes knowledge from both domains
        for i, gat_layer in enumerate(self.gat_layers):
            h = gat_layer(h, edge_index)
            if i < len(self.gat_layers) - 1:
                h = F.elu(h)
                h = self.dropout(h)
        
        # Get user and item embeddings (enriched by cross-domain knowledge)
        user_emb = h[user_indices]
        item_emb = h[item_indices]
        
        # Apply cross-domain fusion
        user_emb = self.cross_domain_fusion(torch.cat([user_emb, user_emb], dim=1))
        item_emb = self.cross_domain_fusion(torch.cat([item_emb, item_emb], dim=1))
        
        # Combine and predict
        combined = torch.cat([user_emb, item_emb], dim=1)
        predictions = self.predictor(combined)
        
        return predictions.squeeze()

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

class CrossDomainTrainer:
    """Enhanced trainer for cross-domain recommendation with comprehensive evaluation"""
    
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
        
        print("Starting cross-domain training...")
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
                    torch.save(self.model.state_dict(), 'best_cross_domain_model.pth')
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print("Early stopping!")
                    break
        
        # Load best model for final evaluation
        self.model.load_state_dict(torch.load('best_cross_domain_model.pth'))
        
        # Final evaluation on test set
        print("\nFinal Cross-Domain Evaluation on Test Set:")
        print("=" * 60)
        
        test_metrics = self.evaluate_per_user(test_data, k_values)
        
        for metric_name, value in test_metrics.items():
            print(f"{metric_name:12s}: {value:.4f}")
        
        return test_metrics

class CrossDomainRecommender:
    """
    Cross-domain recommender that learns from both domains but only recommends All Beauty items
    """
    
    def __init__(self, model, graph_data, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.graph_data = graph_data
        self.device = device
        
    def get_all_beauty_recommendations(self, user_entity, top_k=10):
        """
        Get top-k All Beauty item recommendations for a user
        Uses cross-domain knowledge but only recommends All Beauty items
        """
        self.model.eval()
        
        if user_entity not in self.graph_data.entity_to_new_id:
            print(f"User {user_entity} not found in the graph")
            return []
        
        user_id = self.graph_data.entity_to_new_id[user_entity]
        
        # Get all All Beauty items
        all_beauty_item_entities = list(self.graph_data.all_beauty_items.keys())
        all_beauty_item_ids = [self.graph_data.entity_to_new_id[item] for item in all_beauty_item_entities]
        
        with torch.no_grad():
            x = self.graph_data.x.to(self.device)
            edge_index = self.graph_data.edge_index.to(self.device)
            
            # Create user and item indices for prediction
            user_indices = torch.full((len(all_beauty_item_ids),), user_id, device=self.device)
            item_indices = torch.tensor(all_beauty_item_ids, device=self.device)
            
            # Get predictions
            predictions = self.model(x, edge_index, user_indices, item_indices)
            predictions = torch.sigmoid(predictions)
            
            # Get top-k recommendations
            top_k_values, top_k_indices = torch.topk(predictions, min(top_k, len(predictions)))
            
            recommendations = []
            for i, idx in enumerate(top_k_indices):
                item_entity = all_beauty_item_entities[idx.item()]
                score = top_k_values[i].item()
                recommendations.append((item_entity, score))
        
        return recommendations
    
    def explain_cross_domain_influence(self, user_entity, recommended_item):
        """
        Explain how Beauty & Personal Care knowledge influenced the recommendation
        """
        # This would analyze the attention weights or embeddings to understand
        # how Beauty & PC features influenced the All Beauty recommendation
        explanation = {
            'user': user_entity,
            'recommended_item': recommended_item,
            'cross_domain_features': [],  # Features from Beauty & PC that influenced this
            'explanation': f"Recommendation enriched by Beauty & Personal Care domain knowledge"
        }
        return explanation

def create_user_based_splits_cross_domain(graph_data, train_ratio=0.7, val_ratio=0.15):
    """Create user-based splits for cross-domain data"""
    unique_users = torch.unique(graph_data.user_indices)
    num_users = len(unique_users)
    
    shuffled_indices = torch.randperm(num_users)
    shuffled_users = unique_users[shuffled_indices]
    
    train_user_count = int(train_ratio * num_users)
    val_user_count = int(val_ratio * num_users)
    
    train_users = set(shuffled_users[:train_user_count].tolist())
    val_users = set(shuffled_users[train_user_count:train_user_count + val_user_count].tolist())
    test_users = set(shuffled_users[train_user_count + val_user_count:].tolist())
    
    train_mask = torch.tensor([user.item() in train_users for user in graph_data.user_indices])
    val_mask = torch.tensor([user.item() in val_users for user in graph_data.user_indices])
    test_mask = torch.tensor([user.item() in test_users for user in graph_data.user_indices])
    
    def create_split_data(mask):
        return Data(
            x=graph_data.x,
            edge_index=graph_data.edge_index,
            user_indices=graph_data.user_indices[mask],
            item_indices=graph_data.item_indices[mask],
            labels=graph_data.labels[mask],
            entity_to_new_id=graph_data.entity_to_new_id,
            all_beauty_items=graph_data.all_beauty_items,
            all_beauty_users=graph_data.all_beauty_users
        )
    
    train_data = create_split_data(train_mask)
    val_data = create_split_data(val_mask)
    test_data = create_split_data(test_mask)
    
    print(f"Cross-domain splits:")
    print(f"  Train: {len(train_users)} users, {train_mask.sum().item()} interactions")
    print(f"  Val: {len(val_users)} users, {val_mask.sum().item()} interactions")
    print(f"  Test: {len(test_users)} users, {test_mask.sum().item()} interactions")
    
    return train_data, val_data, test_data

def run_cross_domain_recommendation_system(embeddings_path: str):
    """
    Run the cross-domain recommendation system
    """
    print("=" * 100)
    print("CROSS-DOMAIN RECOMMENDATION SYSTEM")
    print("All Beauty (Target) enriched by Beauty & Personal Care (Source)")
    print("=" * 100)
    
    try:
        # Load domain-specific embeddings
        with open(embeddings_path, 'rb') as f:
            kg_data = pickle.load(f)
        
        # Create cross-domain recommendation data
        graph_data = create_cross_domain_recommendation_data(
            kg_data,
            min_rating_threshold=3.0,
            min_user_interactions=2
        )
        
        if graph_data is None:
            print("Failed to create cross-domain dataset")
            return None
        
        # Create data splits
        train_data, val_data, test_data = create_user_based_splits_cross_domain(graph_data)
        
        # Initialize cross-domain model
        model = CrossDomainGAT(
            num_nodes=len(graph_data.x),
            embedding_dim=64,
            hidden_dim=128,
            num_heads=4,
            num_layers=2,
            dropout=0.3
        )
        
        print(f"Cross-domain model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Initialize enhanced trainer
        trainer = CrossDomainTrainer(model)
        
        # Train and evaluate with comprehensive metrics
        print("\n" + "=" * 60)
        print("TRAINING CROSS-DOMAIN MODEL WITH FULL EVALUATION")
        print("=" * 60)
        
        final_metrics = trainer.train_and_evaluate(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            epochs=500,
            lr=0.001,
            k_values=[5, 10, 20]
        )
        
        # Create recommender for testing
        recommender = CrossDomainRecommender(model, graph_data)
        
        # Test cross-domain recommendations
        print("\n" + "=" * 60)
        print("TESTING CROSS-DOMAIN RECOMMENDATIONS")
        print("=" * 60)
        
        # Get a sample user for testing
        if graph_data.all_beauty_users:
            sample_user = list(graph_data.all_beauty_users.keys())[0]
            print(f"Sample recommendations for user: {sample_user}")
            
            recommendations = recommender.get_all_beauty_recommendations(sample_user, top_k=5)
            
            for i, (item, score) in enumerate(recommendations, 1):
                print(f"  {i}. {item} (score: {score:.4f})")
                
                # Show cross-domain explanation
                explanation = recommender.explain_cross_domain_influence(sample_user, item)
                print(f"     Explanation: {explanation['explanation']}")
        
        print("\n" + "=" * 60)
        print("CROSS-DOMAIN RECOMMENDATION SYSTEM COMPLETE")
        print("=" * 60)
        print("Key Features:")
        print("- Uses knowledge from both All Beauty and Beauty & Personal Care")
        print("- Only recommends All Beauty items")
        print("- Cross-domain knowledge enriches recommendations")
        print("- Comprehensive evaluation with Precision@K, Recall@K, MAP@K, NDCG@K")
        print("- User-based evaluation and data splitting")
        
        print("\n📊 Final Cross-Domain Test Metrics:")
        print("-" * 40)
        for metric_name, value in final_metrics.items():
            print(f"{metric_name}: {value:.4f}")
        
        return model, recommender, final_metrics
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def main_cross_domain():
    """Main function to run cross-domain recommendation"""
    embeddings_path = 'Data/Knowledge graph representations/Domain_Specific_KG_embeddings_clothing.pkl'
    result = run_cross_domain_recommendation_system(embeddings_path)
    
    if result:
        model, recommender, final_metrics = result
        print("\nCross-domain recommendation system ready!")
        print("The model uses Beauty & Personal Care knowledge to enrich All Beauty recommendations.")
        print("Comprehensive evaluation completed with user-based metrics.")
    
    return result

if __name__ == "__main__":
    main_cross_domain()