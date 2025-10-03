import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import torch
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
from pykeen.models import TransD, ComplEx, RotatE,TransR
from pykeen.training import SLCWATrainingLoop
from pykeen.evaluation import RankBasedEvaluator
from pykeen.datasets import Dataset
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
import pickle
import json
import ast
import pickle
import torch
from torch_geometric.data import Data
import numpy as np

import json
import ast
import pandas as pd
from typing import List, Dict, Any, Optional
import sqlite3
from tqdm import tqdm
from Database import SQLiteManager
# Import your existing SQLiteManager class
# from your_database_module import SQLiteManager

import json
import ast
import pandas as pd
from typing import List, Dict, Any, Optional
import sqlite3
from tqdm import tqdm

# Import your existing SQLiteManager class
# from your_database_module import SQLiteManager

import json
import ast
import pandas as pd
from typing import List, Dict, Any, Optional
import sqlite3
from tqdm import tqdm

# Import your existing SQLiteManager class
# from your_database_module import SQLiteManager

import json
import ast
import pandas as pd
from typing import List, Dict, Any, Optional
import sqlite3
from tqdm import tqdm

# Import your existing SQLiteManager class
# from your_database_module import SQLiteManager

import json
import ast
import pandas as pd
from typing import List, Dict, Any, Optional
import sqlite3
from tqdm import tqdm

# Import your existing SQLiteManager class
# from your_database_module import SQLiteManager

def extract_triplets_from_database(db_path: str,
                                 source_table: str,
                                 target_table: str = "knowledge_triplets") -> int:
    """
    Extract knowledge graph triplets from SQLite table and insert into another table.
    Creates independent SQLiteManager objects to work with the database.
    
    Parameters:
    db_path (str): Path to the SQLite database file
    source_table (str): Name of the source table containing review data
    target_table (str): Name of the target table to store triplets
    
    Returns:
    int: Number of triplets inserted
    """
    
    # Create SQLiteManager instance
    db_manager = SQLiteManager(db_path)
    
    try:
        db_manager.connect()
        
        # Step 1: Drop and recreate target table
        create_triplets_table(db_manager, target_table)
        
        # Step 2: Read all data at once
        query = f"""
        SELECT user_id, parent_asin, llm_enrichment, rating
        FROM {source_table}
        WHERE llm_enrichment IS NOT NULL 
        AND llm_enrichment != ''
        ORDER BY rowid
        """
        
        print(f"Reading all data from {source_table}...")
        all_data = db_manager.select_command_executer(query)
        print(f"Retrieved {len(all_data):,} records")
        
        # Step 3: Process all data and extract triplets
        print("Processing data and extracting triplets...")
        triplets = process_all_data_for_triplets(all_data)
        
        # Step 4: Insert all triplets at once
        if triplets:
            print(f"Inserting {len(triplets):,} triplets into {target_table}...")
            total_triplets = insert_all_triplets(db_manager, target_table, triplets)
        else:
            total_triplets = 0
        
        print(f"Successfully extracted {total_triplets:,} triplets into {target_table}")
        return total_triplets
        
    except Exception as e:
        print(f"Error during triplet extraction: {e}")
        raise
    finally:
        db_manager.disconnect()


def create_triplets_table(db_manager, table_name: str):
    """
    Create the triplets table using SQLiteManager's create_table method.
    Includes head, relation, tail plus source data columns and timestamp.
    """
    headers = ['head', 'relation', 'tail', 'user_id', 'parent_asin', 'llm_enrichment', 'timestamp']
    column_types = ['TEXT', 'TEXT', 'TEXT', 'TEXT', 'TEXT', 'TEXT', 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP']
    
    # Use existing create_table method
    db_manager.create_table(table_name, headers, column_types, drop_if_exists=True)
    
    # Add unique constraint on the triplet columns only
    cursor = db_manager.connection.cursor()
    try:
        cursor.execute(f"""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_{table_name}_unique 
        ON {table_name}(head, relation, tail)
        """)
        db_manager.connection.commit()
        print(f"Created unique constraint on {table_name}")
    except sqlite3.Error as e:
        print(f"Note: Unique constraint creation: {e}")


def process_all_data_for_triplets(all_data: List[tuple]) -> List[Dict[str, str]]:
    """
    Process all data and extract triplets.
    Handles the new llm_enrichment format with phrase keys and [category, sentiment] arrays.
    """
    
    # Convert tuples to DataFrame for easier processing
    df = pd.DataFrame(all_data, columns=['user_id', 'parent_asin', 'llm_enrichment', 'rating'])
    
    # Create coded versions (same logic as original)
    df["user_id_coded"] = pd.Categorical(df["user_id"]).codes 
    df["user_id_coded"] = df["user_id_coded"].astype(str).apply(lambda x: "user_" + x)
    df["parent_asin_coded"] = pd.Categorical(df["parent_asin"]).codes 
    df["parent_asin_coded"] = df["parent_asin_coded"].astype(str).apply(lambda x: "item_" + x)
    
    triplets = []
    
    for _, row in df.iterrows():
        user_id = row['user_id_coded']
        parent_asin = row['parent_asin_coded']
        rating = str(row['rating'])
        llm_enrichment = row['llm_enrichment']
        
        # Parse llm_enrichment (updated for problematic format)
        try:
            if isinstance(llm_enrichment, str):
                # Clean up the string first
                cleaned_str = llm_enrichment.strip()
                
                # More robust approach: parse as Python dictionary literal
                # Step 1: Fix the outer brackets to make it a dict
                if cleaned_str.startswith('[') and cleaned_str.endswith(']'):
                    # Extract the content between outer brackets
                    content = cleaned_str[1:-1]
                    # Wrap in curly braces to make it a dictionary
                    cleaned_str = '{' + content + '}'
                
                # Step 2: Fix double quotes inside strings (""text"" -> "text")
                import re
                cleaned_str = re.sub(r'""([^"]*?)""', r'"\1"', cleaned_str)
                
                # Step 3: Remove + signs from positive numbers
                cleaned_str = re.sub(r'\+(\d+\.?\d*)', r'\1', cleaned_str)
                
                print(f"Cleaned string: {cleaned_str}")  # Debug output
                
                try:
                    # Try parsing the cleaned string as JSON
                    analysis_dict = json.loads(cleaned_str)
                except json.JSONDecodeError as json_err:
                    print(f"JSON parsing failed: {json_err}")
                    # If JSON still fails, try ast.literal_eval
                    try:
                        analysis_dict = ast.literal_eval(cleaned_str)
                    except (ValueError, SyntaxError) as ast_err:
                        print(f"AST parsing also failed: {ast_err}")
                        raise ast_err
            else:
                analysis_dict = llm_enrichment
                
            # Ensure it's a dictionary
            if not isinstance(analysis_dict, dict):
                continue
                
        except (ValueError, SyntaxError, json.JSONDecodeError) as e:
            print(f"Error parsing llm_enrichment for user {user_id}: {e}")
            print(f"Raw data: {llm_enrichment}")
            continue
        
        # Add the rating triplet for each row
        triplets.append({
            'head': user_id,
            'relation': f"rates_{rating}",
            'tail': parent_asin,
            'user_id': row['user_id'],
            'parent_asin': row['parent_asin'],
            'llm_enrichment': llm_enrichment
        })
        
        # Process each key-value pair in the llm_enrichment dictionary
        for phrase, analysis_array in analysis_dict.items():
            if not isinstance(analysis_array, list) or len(analysis_array) != 2:
                continue
                
            # Extract category and sentiment from the array
            category = analysis_array[0]
            sentiment_value = analysis_array[1]
            
            # Clean up category (remove extra quotes if present)
            if isinstance(category, str):
                category = category.strip().strip('"').strip("'")
            
            if not category:
                continue
            
            try:
                sentiment = float(sentiment_value)
            except (ValueError, TypeError):
                continue
            
            # Create triplets based on sentiment
            if sentiment > 0:
                triplets.append({
                    'head': user_id,
                    'relation': 'likes',
                    'tail': "feature_" + category,
                    'user_id': row['user_id'],
                    'parent_asin': row['parent_asin'],
                    'llm_enrichment': llm_enrichment
                })
                
                triplets.append({
                    'head': parent_asin,
                    'relation': 'has',
                    'tail': "feature_" + category,
                    'user_id': row['user_id'],
                    'parent_asin': row['parent_asin'],
                    'llm_enrichment': llm_enrichment
                })
                
            elif sentiment < 0:
                triplets.append({
                    'head': user_id,
                    'relation': 'dislikes',
                    'tail': "feature_" + category,
                    'user_id': row['user_id'],
                    'parent_asin': row['parent_asin'],
                    'llm_enrichment': llm_enrichment
                })
                
                triplets.append({
                    'head': parent_asin,
                    'relation': 'has',
                    'tail': "feature_" + category,
                    'user_id': row['user_id'],
                    'parent_asin': row['parent_asin'],
                    'llm_enrichment': llm_enrichment
                })
    
    return triplets


def insert_all_triplets(db_manager, table_name: str, triplets: List[Dict[str, str]]) -> int:
    """
    Insert all triplets into the target table at once.
    Uses raw SQL for better performance.
    """
    
    if not triplets:
        return 0
    
    # Remove duplicates within the triplets (based on head, relation, tail only)
    df_triplets = pd.DataFrame(triplets)
    df_triplets = df_triplets.drop_duplicates(subset=['head', 'relation', 'tail']).reset_index(drop=True)
    
    cursor = db_manager.connection.cursor()
    
    try:
        # Use INSERT OR IGNORE to handle duplicates
        insert_sql = f"""
        INSERT OR IGNORE INTO {table_name} (head, relation, tail, user_id, parent_asin, llm_enrichment) 
        VALUES (?, ?, ?, ?, ?, ?)
        """
        
        # Convert DataFrame to list of tuples
        triplet_tuples = [
            (row['head'], row['relation'], row['tail'], row['user_id'], row['parent_asin'], row['llm_enrichment']) 
            for _, row in df_triplets.iterrows()
        ]
        
        # Execute batch insert
        cursor.executemany(insert_sql, triplet_tuples)
        db_manager.connection.commit()
        
        return len(triplet_tuples)
        
    except sqlite3.Error as e:
        print(f"Error inserting triplets: {e}")
        raise

def load_triplets_to_dataframe(db_path: str, 
                              table_name: str = "knowledge_triplets",
                              limit: Optional[int] = None) -> pd.DataFrame:
    """
    Load head, relation, tail columns from triplets table into a pandas DataFrame.
    
    Parameters:
    db_path (str): Path to the SQLite database file
    table_name (str): Name of the triplets table
    limit (int, optional): Limit number of rows to load
    
    Returns:
    pd.DataFrame: DataFrame with head, relation, tail columns
    """
    
    # Create SQLiteManager instance
    db_manager = SQLiteManager(db_path)
    
    try:
        db_manager.connect()
        
        # Build query to select only head, relation, tail columns
        if limit:
            query = f"SELECT head, relation, tail FROM {table_name} LIMIT {limit}"
        else:
            query = f"SELECT head, relation, tail FROM {table_name}"
        
        print(f"Loading head, relation, tail from {table_name} into DataFrame...")
        
        # Execute query and get results
        results = db_manager.select_command_executer(query)
        
        # Convert to DataFrame
        df = pd.DataFrame(results, columns=['head', 'relation', 'tail'])
        
        print(f"Successfully loaded {len(df):,} triplets into DataFrame")
        
        return df
        
    except Exception as e:
        print(f"Error loading triplets to DataFrame: {e}")
        raise
    finally:
        db_manager.disconnect()




class KnowledgeGraphGenerator:
    """
    A class to generate and train knowledge graphs from CSV data using PyKEEN.
    
    Expected CSV format:
    - Columns: subject, relation, object
    - Relations: 'likes', 'dislikes', 'rates_X', 'has'
    - Nodes: users (user_*), items (item_*), features (feature_*)
    """
    
    def __init__(self):
        self.triples_factory = None
        self.model = None
        self.training_results = None
        self.entity_to_id = None
        self.relation_to_id = None
        self.embeddings_dim = 64
        
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
            print(f"CSV columns: {df.columns.tolist()}")
            # Validate required columns
            required_columns = ['head', 'relation', 'tail']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"CSV must contain columns: {required_columns}")
            
            print(f"Loaded {len(df)} triples from CSV")
            print(f"Unique entities: {len(set(df['head'].unique()) | set(df['tail'].unique()))}")
            print(f"Unique relations: {df['relation'].nunique()}")
            
            return df
            
        except Exception as e:
            raise Exception(f"Error loading CSV data: {str(e)}")
    
    def create_triples_factory(self, df: pd.DataFrame) -> TriplesFactory:
        """Create triples factory from DataFrame"""
        # Store mappings
        self.entity_to_id = {}
        self.id_to_entity = {}
        self.relation_to_id = {}
        
        # Convert DataFrame to array of triples
        triples = df[['head', 'relation', 'tail']].values
        
        # Create entity and relation mappings
        unique_entities = pd.concat([df['head'], df['tail']]).unique()
        unique_relations = df['relation'].unique()
        
        for idx, entity in enumerate(unique_entities):
            self.entity_to_id[entity] = idx
            self.id_to_entity[idx] = entity
        
        for idx, relation in enumerate(unique_relations):
            self.relation_to_id[relation] = idx
    
        # Create triples factory
        self.triples_factory = TriplesFactory(
            mapped_triples=torch.tensor([
                [self.entity_to_id[h], self.relation_to_id[r], self.entity_to_id[t]]
                for h, r, t in triples
            ]),
            entity_to_id=self.entity_to_id,
            relation_to_id=self.relation_to_id,
            create_inverse_triples=True
        )
        return self.triples_factory
    
    def train_knowledge_graph(self, training: TriplesFactory, model_name: str, epochs: int, 
                        embedding_dim: int, learning_rate: float) -> dict:
        """
        Train a knowledge graph embedding model using PyKEEN on all available data.
        
        Args:
            training: Training triples factory containing all triples
            model_name: Name of the embedding model ('TransE', 'ComplEx', 'RotatE')
            epochs: Number of training epochs
            embedding_dim: Dimension of entity/relation embeddings
            learning_rate: Learning rate for optimization
            
        Returns:
            Dictionary containing training results
        """
        print(f"Training {model_name} model on all available data...")
        
        # Configure model parameters
        model_kwargs = {
            'embedding_dim': embedding_dim,
        }
        # self.triples_factory = training
        
        # Model-specific parameters
        if model_name == 'ComplEx':
            model_kwargs['embedding_dim'] = embedding_dim // 2  # ComplEx uses complex embeddings
        
        try:
            # Run the training pipeline - use same data for training and testing to avoid splits
            results = pipeline(
                training=self.triples_factory,
                testing=self.triples_factory,  # Use same data to avoid creating internal splits
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
            entity_embeddings = results.model.entity_representations[0]().cpu().detach().numpy()
            print(entity_embeddings.shape)
            print(entity_embeddings[0])
            self.model = results.model
            self.training_results = results
            
            # Print learning quality metrics
            print("\n" + "="*50)
            print("LEARNING QUALITY METRICS")
            print("="*50)
            
            if results.losses and len(results.losses) > 0:
                initial_loss = results.losses[0]
                final_loss = results.losses[-1]
                
                print(f"Initial loss (epoch 1): {initial_loss:.4f}")
                print(f"Final loss (epoch {epochs}): {final_loss:.4f}")
                
                # Calculate improvement
                loss_reduction = initial_loss - final_loss
                loss_reduction_percent = (loss_reduction / initial_loss) * 100 if initial_loss > 0 else 0
                
                print(f"Loss reduction: {loss_reduction:.4f} ({loss_reduction_percent:.1f}%)")
                
        
        except Exception as e:
            raise Exception(f"Error during training: {str(e)}")     
        
        return results            
    
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

    def _categorize_entities(self):
        """Categorize entities into users, items, and features based on entity names."""
        
        if self.entity_to_id is None:
            raise ValueError("entity_to_id mapping not available. Train the model first.")
        
        self.user_entities = {}
        self.item_entities = {}
        self.feature_entities = {}
        
        for entity, entity_id in self.entity_to_id.items():
            if entity.startswith('user_'):
                self.user_entities[entity] = entity_id
            elif entity.startswith('item_'):
                self.item_entities[entity] = entity_id
            elif entity.startswith('feature_'):
                self.feature_entities[entity] = entity_id
        
        print(f"Categorized entities:")
        print(f"  Users: {len(self.user_entities)}")
        print(f"  Items: {len(self.item_entities)}")
        print(f"  Features: {len(self.feature_entities)}")

    def _get_string_triples(self):
        """Convert mapped triples back to string format."""
        string_triples = []
        mapped_triples = self.triples_factory.mapped_triples.cpu().numpy()
        
        for triple in mapped_triples:
            head_id, relation_id, tail_id = triple
            head = self.id_to_entity[head_id]
            relation = {v: k for k, v in self.relation_to_id.items()}[relation_id]
            tail = self.id_to_entity[tail_id]
            string_triples.append([head, relation, tail])
        
        return string_triples
            
    def fetch_embeddings_for_gnn(self, save_path: Optional[str] = None) -> dict:
        """Fetch embeddings using stored mappings"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        print("Fetching embeddings for GNN...")
        
        # Ensure entities are categorized
        if not hasattr(self, 'user_entities') or self.user_entities is None:
            print("Categorizing entities...")
            self._categorize_entities()


        embeddings_data = {
            'entity_embeddings': self.get_entity_embeddings().cpu().numpy(),
            'relation_embeddings': self.get_relation_embeddings().cpu().numpy(),
            'entity_to_id': self.entity_to_id,
            'relation_to_id': self.relation_to_id,
            'id_to_entity': self.id_to_entity,
            'id_to_relation': {v: k for k, v in self.relation_to_id.items()},
            'embedding_dim': self.get_entity_embeddings().shape[1],  # Get actual dimension
            'num_entities': len(self.entity_to_id),
            'num_relations': len(self.relation_to_id),
            'user_entities': self.user_entities,
            'item_entities': self.item_entities,
            'feature_entities': self.feature_entities,
            'num_users': len(self.user_entities),
            'num_items': len(self.item_entities),
            'num_features': len(self.feature_entities),
            'triples': self._get_string_triples()
        }
        
        print(f"Fetched embeddings for {len(self.entity_to_id)} entities and {len(self.relation_to_id)} relations.")    
        print(f"Entity embeddings shape: {embeddings_data['entity_embeddings'].shape}")
        print(f"Relation embeddings shape: {embeddings_data['relation_embeddings'].shape}")
        print(f"Users: {len(self.user_entities)}")
        print(f"Items: {len(self.item_entities)}")
        print(f"Features: {len(self.feature_entities)}")
        
        if save_path:
            with open(save_path, 'wb') as f:
                pickle.dump(embeddings_data, f)
            print(f"Embeddings saved to {save_path}")
        
        return embeddings_data
    
# Example usage function
def main():
    """Example usage of the KnowledgeGraphGenerator."""
    
    # Initialize the generator
    kg_generator = KnowledgeGraphGenerator()

    # Extract triplets from your review data
    total_triplets = extract_triplets_from_database(
        db_path="Amazon_Review.db",
        source_table="Clothing_Common_Users",
        target_table="Clothing_Common_Users_KG_triplets"
    )
    triplets_df = load_triplets_to_dataframe(
        db_path="Amazon_Review.db",
        table_name="Clothing_Common_Users_KG_triplets",
        #limit=10000  # Optional: limit for testing
    )
    
    kg_generator.create_triples_factory(triplets_df)
    
    # Train the model (now with testing parameter)
    results = kg_generator.train_knowledge_graph(
        kg_generator.triples_factory,
        model_name='TransR', # TransR and TransD are the best so far
        epochs=100, #700 is the best so far
        embedding_dim=64,
        learning_rate = 0.001 #0.001 is the best so far
    )
    
    ## Visualize embeddings
    # print("\n" + "="*60)
    # print("VISUALIZATION")
    # print("="*60)
    # kg_generator.visualize_embeddings()
    
    print("\n" + "="*60)
    print("SAVING EMBEDDINGS FOR GNN")
    print("="*60)
    
    # OR if you added it as a class method:
    embeddings_data = kg_generator.fetch_embeddings_for_gnn(save_path='Data/Knowledge graph representations/KG_embeddings_for_gnn_cloth.pkl')
    
    return kg_generator, embeddings_data  # Return both
  

if __name__ == "__main__":
    # First extract triplets from the CSV
    # triplets_df = extract_triplets_from_csv('Data/LLM Output sampels/LLM_result_All_Beauty_V2_F50.csv','Data/Knowledge graph extracted triplets/All_Beauty.csv')\
    # # if we have the triplests already, we can skip this step and just load the
    # triples_df = pd.read_csv('Data/Knowledge graph extracted triplets/generated.csv')
    main()  # Now returns embeddings too
    
    print("\n" + "="*60)
    print("READY FOR GNN TRAINING")
    print("="*60)
    print("Embeddings saved to: embeddings_for_gnn.pkl")
    print("Now you can run the GNN recommender system!")



