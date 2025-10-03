import pandas as pd
import torch
import json
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data

class GATGraphBuilder:
    def __init__(self, embedding_dim=64):
        self.embedding_dim = embedding_dim
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.graph_data = None
        
    def load_csv(self, csv_path, user_col='user_id', item_col='parent_asin', rating_col='rating'):
        """Load CSV file with user-item ratings."""
        self.df = pd.read_csv(csv_path)
        self.df['rating'] = self.df['rating'].astype(int)
        
        # Check if columns exist
        required_cols = [user_col, item_col, rating_col]
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        # Rename columns for consistency
        self.df = self.df.rename(columns={
            user_col: 'user',
            item_col: 'item', 
            rating_col: 'rating'
        })
        self.df = self.df[['user', 'item', 'rating']]
        
        # Remove duplicates and missing values
        self.df = self.df.dropna().drop_duplicates(subset=['user', 'item'])
        
        print(f"Loaded {len(self.df)} user-item interactions")
        print(f"Unique users: {self.df['user'].nunique()}")
        print(f"Unique items: {self.df['item'].nunique()}")
        
    def build_graph_for_gat(self):
        """Build bipartite graph specifically for GAT input."""
        
        # Encode users and items to integer IDs
        unique_users = self.df['user'].unique()
        unique_items = self.df['item'].unique()
        
        self.user_encoder.fit(unique_users)
        self.item_encoder.fit(unique_items)
        
        n_users = len(unique_users)
        n_items = len(unique_items)
        n_nodes = n_users + n_items
        
        # Build edge list for GAT
        edge_index = []
        edge_weights = []
        
        for _, row in self.df.iterrows():
            user_id = self.user_encoder.transform([row['user']])[0]
            item_id = self.item_encoder.transform([row['item']])[0] + n_users
            weight = row['rating']
            
            # Add bidirectional edges
            edge_index.extend([[user_id, item_id], [item_id, user_id]])
            edge_weights.extend([weight, weight])
            
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_weights = torch.tensor(edge_weights, dtype=torch.float)
        
        # Initialize node features (random embeddings for GAT)
        node_features = torch.randn(n_nodes, self.embedding_dim)
        
        # Create node type labels (0 for users, 1 for items)
        node_types = torch.cat([
            torch.zeros(n_users, dtype=torch.long),  # Users
            torch.ones(n_items, dtype=torch.long)    # Items
        ])
        
        # Create PyTorch Geometric Data object for GAT
        self.graph_data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_weights.unsqueeze(1),
            node_type=node_types,
            num_users=n_users,
            num_items=n_items
        )
        
        print(f"Graph ready for GAT - Nodes: {n_nodes}, Edges: {len(edge_weights)}")
        return self.graph_data
    
    def export_for_gat(self, filepath):
        """Export graph in format ready for GAT training."""
        if self.graph_data is None:
            raise ValueError("Build graph first using build_graph_for_gat()")
            
        gat_data = {
            'node_features': self.graph_data.x.numpy().tolist(),
            'edge_index': self.graph_data.edge_index.numpy().tolist(),
            'edge_weights': self.graph_data.edge_attr.squeeze().numpy().tolist(),
            'node_types': self.graph_data.node_type.numpy().tolist(),
            'num_users': int(self.graph_data.num_users),
            'num_items': int(self.graph_data.num_items),
            'embedding_dim': self.embedding_dim,
            'user_labels': self.user_encoder.classes_.tolist(),
            'item_labels': self.item_encoder.classes_.tolist()
        }
        
        with open(filepath, 'w') as f:
            json.dump(gat_data, f, indent=2)
        print(f"GAT-ready data exported to {filepath}")
        
    def get_gat_inputs(self):
        """Get direct inputs for GAT model."""
        if self.graph_data is None:
            raise ValueError("Build graph first using build_graph_for_gat()")
            
        return {
            'x': self.graph_data.x,  # Node features
            'edge_index': self.graph_data.edge_index,  # Edge connectivity
            'edge_attr': self.graph_data.edge_attr,  # Edge weights/features
            'node_type': self.graph_data.node_type,  # Node types (user/item)
            'num_nodes': self.graph_data.x.size(0)
        }

def main():
    # Create GAT graph builder
    builder = GATGraphBuilder(embedding_dim=64)
    
    try:
        # Load CSV data
        builder.load_csv('Data/LLM Output sampels/LLM_result_All_Beauty_V2_F50.csv', 
                        user_col='user_id', 
                        item_col='parent_asin', 
                        rating_col='rating')
        
        # Build graph for GAT
        graph_data = builder.build_graph_for_gat()
        
        # Export for GAT training
        builder.export_for_gat('graph_for_gat.json')
        
        # Get direct GAT inputs
        gat_inputs = builder.get_gat_inputs()
        print(f"\nGAT Input shapes:")
        print(f"Node features (x): {gat_inputs['x'].shape}")
        print(f"Edge index: {gat_inputs['edge_index'].shape}")
        print(f"Edge attributes: {gat_inputs['edge_attr'].shape}")
        print(f"Node types: {gat_inputs['node_type'].shape}")
        
    except FileNotFoundError:
        print("CSV file not found. Please provide the correct path to your CSV file.")

if __name__ == "__main__":
    main()