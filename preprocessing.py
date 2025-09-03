import pandas as pd
import traceback
import json
from typing import List, Union, Iterator
import gzip
import os

""" This file contains functions for identifying common users across high-resource and low-resource domains.  
- **fetch_common_users_from_high_resource_domain**: Extracts common users from the high-resource domain and saves them as a CSV file.  
- **fetch_common_users_from_low_resource_domain**: Computes the common users in the low-resource domain, based on the filtered users obtained from the high-resource domain.  
- **number_of_reviews**: Counts the number of occurrences of a specific JSON tag in a .jsonl.gz file.
- **divide_low_resource_dataset_for_llm_enrichment**: Divides the low-resource dataset into three parts: users unique to the first dataset, users unique to the second dataset, and users common to both datasets. Each part is saved as a separate CSV file.
"""  

def number_of_reviews(file_path, tag_name):
    """
    Count occurrences of a specific JSON tag in a jsonl.gz file.
    
    Args:
        file_path (str): Path to the .jsonl.gz file
        tag_name (str): Name of the JSON tag to count
        
    Returns:
        int: Number of occurrences of the tag
    """
    count = 0
    
    with gzip.open(file_path, 'rt', encoding='utf-8') as file:
        for line in file:
            try:
                json_obj = json.loads(line.strip())
                if tag_name in json_obj:
                    count += 1
            except json.JSONDecodeError:
                # Skip invalid JSON lines
                continue
    
    return count

def fetch_common_users_from_high_resource_domain(folder_path_high_resource_domain, file_path_low_resource_domain):
    """
    Read two JSONL files (From a high resource domain), merge them, filter the users (from a low resource) and save as CSV.
    
    Args:
        folder_path_high_resource_domain (str): Path to the directory containing the files
        file_path_low_resource_domain (str): Path to low resource domain file
        batch_size (int): Batch size for processing
    """
    print(f"DEBUG: Starting processing...")
    print(f"DEBUG: High resource folder: {folder_path_high_resource_domain}")
    print(f"DEBUG: Low resource file: {file_path_low_resource_domain}")
    
    # Check if paths exist
    if not os.path.exists(folder_path_high_resource_domain):
        print(f"ERROR: High resource folder does not exist: {folder_path_high_resource_domain}")
        return
        
    if not os.path.exists(file_path_low_resource_domain):
        print(f"ERROR: Low resource file does not exist: {file_path_low_resource_domain}")
        return

    meta_file = None
    main_file = None
    print(f"Processing folder: {folder_path_high_resource_domain}")
    
    # Find the two files - main and meta
    files_in_folder = os.listdir(folder_path_high_resource_domain)
    print(f"DEBUG: Files in folder: {files_in_folder}")
    
    for item in files_in_folder:
        if item.endswith('.jsonl'):
            print(f"Found JSONL file: {item}")
            if 'meta' in item.lower():
                meta_file = item
                print(f"  -> Identified as meta file")
            else:
                main_file = item
                print(f"  -> Identified as main file")
    
    if not main_file or not meta_file:
        print(f"ERROR: Could not find both main and meta files")
        print(f"  Main file: {main_file}")
        print(f"  Meta file: {meta_file}")
        return
    
    # Load user IDs from low resource domain
    print("Loading user IDs from low resource domain...")
    try:
        low_resource_df = pd.read_json(file_path_low_resource_domain, lines=True)
        user_id_set = set(low_resource_df['user_id'].tolist())
        print(f"Loaded {len(user_id_set)} user IDs from low resource domain.")
        print(f"Sample user IDs: {list(user_id_set)[:5]}")
    except Exception as e:
        print(f"ERROR loading low resource file: {e}")
        return
    
    # Filter main file of high resource domain based on user IDs in low resource domain
    main_file_path = os.path.join(folder_path_high_resource_domain, main_file)
    print(f"Processing main file: {main_file_path}")
    
    # Check file size
    file_size = os.path.getsize(main_file_path) / (1024 * 1024)
    print(f"Main file size: {file_size:.2f} MB")
    
    all_filtered_entries = []
    count = 0
    matched_count = 0
    
    try:
        with open(main_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                count += 1
                
                # Progress update every 50k lines
                if count % 1000000 == 0:
                    print(f"Processed {count} lines, found {matched_count} matches")
                
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        user_id = entry.get('user_id')
                        if user_id in user_id_set:
                            matched_count += 1
                            all_filtered_entries.append(entry)
                    
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error at line {count}: {e}")
                        continue
        
        print(f"Finished processing main file:")
        print(f"  Total lines processed: {count}")
        print(f"  Total matches found: {matched_count}")
        
        if matched_count == 0:
            print("WARNING: No matching users found!")
            return
        
        # Convert to DataFrame
        df_main_filtered = pd.DataFrame(all_filtered_entries)
        print(f"Main DataFrame shape after filtering: {df_main_filtered.shape}")
        print(f"Main DataFrame columns: {df_main_filtered.columns.tolist()}")
        
        # Create review column
        if 'title' in df_main_filtered.columns and 'text' in df_main_filtered.columns:
            df_main_filtered['review'] = df_main_filtered['title'].fillna('') + " " + df_main_filtered['text'].fillna('')
            df_main_filtered.drop(columns=['title', 'text'], inplace=True)
            print("Created 'review' column and dropped 'title' and 'text'")
        else:
            print("WARNING: 'title' or 'text' columns not found")
    except Exception as e:
            print(f"ERROR processing main file: {e}")
            
            traceback.print_exc()
            return
    ######################################################
# Load meta file line by line and filter by parent_asin
    meta_file_path = os.path.join(folder_path_high_resource_domain, meta_file)
    print(f"Loading meta file: {meta_file_path}")
        
    # Get file size for progress tracking
    meta_file_size = os.path.getsize(meta_file_path) / (1024 * 1024)
    print(f"Meta file size: {meta_file_size:.2f} MB")
        
    # Create set of parent_asin values from the filtered main data
    parent_asin_set = set(df_main_filtered['parent_asin'].dropna().tolist()) if 'parent_asin' in df_main_filtered.columns else set()
    print(f"Looking for {len(parent_asin_set)} unique parent_asin values in meta file")
    print(f"Sample parent_asin values: {list(parent_asin_set)[:5]}")
        
    if not parent_asin_set:
        print("ERROR: No parent_asin values found in main data - cannot merge")
        return
        
    try:
        all_meta_entries = []
        meta_count = 0
        meta_matched_count = 0
            
        with open(meta_file_path, 'r', encoding='utf-8') as meta_file_handle:
            for line in meta_file_handle:
                meta_count += 1
                    
                # Progress update every 100k lines for meta file
                if meta_count % 100000 == 0:
                    print(f"Processed {meta_count} meta lines, found {meta_matched_count} matches")
                    
                line = line.strip()
                if line:
                    try:
                        meta_entry = json.loads(line)
                            
                        # Check if this meta entry's parent_asin is in our filtered main data
                        meta_parent_asin = meta_entry.get('parent_asin')
                        if meta_parent_asin in parent_asin_set:
                            meta_matched_count += 1
                            all_meta_entries.append(meta_entry)
                                
                            # Debug: Show first few meta matches
                            if meta_matched_count <= 3:
                                   print(f"META MATCH {meta_matched_count}: Found parent_asin {meta_parent_asin}")
                        
                    except json.JSONDecodeError as e:
                        if meta_count <= 10:  # Only show first few meta decode errors
                            print(f"Meta JSON decode error at line {meta_count}: {e}")
                        continue
            
        print(f"Finished processing meta file:")
        print(f"  Total meta lines processed: {meta_count}")
        print(f"  Total meta matches found: {meta_matched_count}")
            
        if meta_matched_count == 0:
            print("WARNING: No matching parent_asin found in meta file!")
            return
            
        # Convert meta entries to DataFrame
        df_meta = pd.DataFrame(all_meta_entries)
        print(f"Meta DataFrame shape after filtering: {df_meta.shape}")
        print(f"Meta DataFrame columns: {df_meta.columns.tolist()}")
            
        # Rename title column in meta to avoid conflicts
        if 'title' in df_meta.columns:
            df_meta.rename(columns={'title': 'product_title'}, inplace=True)
            print("Renamed 'title' to 'product_title' in meta data")
            
        # Merge dataframes
        merge_column = 'parent_asin'
        if merge_column in df_main_filtered.columns and merge_column in df_meta.columns:
            print(f"Merging on '{merge_column}' column...")
            merged_df = df_main_filtered.merge(df_meta, on=merge_column, how='left')
            print(f"DataFrame shape after merging: {merged_df.shape}")
            print(f"All columns after merge: {merged_df.columns.tolist()}")
                
            # Check merge success
            null_merges = merged_df['product_title'].isnull().sum() if 'product_title' in merged_df.columns else 0
            print(f"Rows with missing product info after merge: {null_merges}")
                
            # Save as CSV
            output_file = os.path.join(folder_path_high_resource_domain, 'merged_and_filtered_based_on_users.csv')
            merged_df.to_csv(output_file, index=False)
            print(f"SUCCESS: Merged data saved to: {output_file}")
            print(f"Final dataset shape: {merged_df.shape}")
        else:
            print(f"ERROR: Cannot merge - '{merge_column}' column missing")
            print(f"  Main columns: {df_main_filtered.columns.tolist()}")
            print(f"  Meta columns: {df_meta.columns.tolist()}")
                
    except Exception as e:
        print(f"ERROR processing meta file: {e}")
        import traceback
        traceback.print_exc()
        return
        
def fetch_common_users_from_low_resource_domain(file_path_low_resource_domain, filtered_users_file_from_high_resource):
    """
    Read a csv file (From a low resource domain), filter the users based on a provided CSV of user IDs, and save as CSV.
    
    Args:
        file_path_low_resource_domain (str): Path to the file containing the low resource merged data
        filtered_users_file_from_high_resource (str): Path to CSV file containing filtered user IDs of the high resource domain
    """
    print(f"DEBUG: Starting processing for low resource domain...")
    print(f"DEBUG: Low resource folder: {file_path_low_resource_domain}") ## this file contains the merged data of low resource domain
    print(f"DEBUG: Filtered users file: {filtered_users_file_from_high_resource}")
    
    # Check if paths exist
    if not os.path.exists(file_path_low_resource_domain):
        print(f"ERROR: Low resource folder does not exist: {file_path_low_resource_domain}")
        return
        
    if not os.path.exists(filtered_users_file_from_high_resource):
        print(f"ERROR: Filtered users file does not exist: {filtered_users_file_from_high_resource}")
        return
    # Load low resource domain data
    print(f"Processing file: {file_path_low_resource_domain}")
    low_resource_df = pd.read_csv(file_path_low_resource_domain)
    
    # Load user IDs from filtered users CSV
    print("Loading filtered user IDs...")
    try:
        filtered_users_high_resource = pd.read_csv(filtered_users_file_from_high_resource, usecols=['user_id'])
        print(f"Entries in filtered high resource dataFrame: {filtered_users_high_resource.shape[0]}")
        user_id_unique_high_resource = set(filtered_users_high_resource['user_id'].tolist())
        print(f"Unique user IDs in filtered high resource dataset:", len(user_id_unique_high_resource))
        print(f"Sample user IDs: {list(user_id_unique_high_resource)[:5]}")
    except Exception as e:
        print(f"ERROR loading filtered users file: {e}")
        return
    
    # Filter low resource file based on user IDs
    low_resource_df_filtered = low_resource_df[low_resource_df['user_id'].isin(user_id_unique_high_resource)]
    print(f"Low resource DataFrame shape after filtering: {low_resource_df_filtered.shape}")
    print(len(set(low_resource_df_filtered['user_id'].tolist())), "unique users in low resource domain before filtering")
    low_resource_df_filtered.to_csv(os.path.join(os.path.dirname(file_path_low_resource_domain), os.path.basename(file_path_low_resource_domain)+os.path.basename(filtered_users_file_from_high_resource)+'.csv'), index=False)

def divide_low_resource_dataset_for_llm_enrichment(file_path_common_users_1, file_path_common_users_2):
    
    df_common_users_1 = pd.read_csv(file_path_common_users_1, usecols=['user_id'])
    df_common_users_2 = pd.read_csv(file_path_common_users_2, usecols=['user_id'])
    print(len(df_common_users_1), "entries in common users dataset 1", df_common_users_1['user_id'].nunique(), "unique users")
    print(len(df_common_users_2), "entries in common users dataset 2", df_common_users_2['user_id'].nunique(), "unique users")
    
    user_df_common_users_1 = set(df_common_users_1['user_id'].tolist())
    user_df_common_users_2 = set(df_common_users_2['user_id'].tolist())

    common_users = user_df_common_users_1.intersection(user_df_common_users_2)
    print(len(common_users), "common unique users between two datasets")
    user_1 = user_df_common_users_1 - common_users
    user_2 = user_df_common_users_2 - common_users
    df_common_users_1_filtered = df_common_users_1[df_common_users_1['user_id'].isin(user_1)]
    print(len(df_common_users_1_filtered), "entries in filtered common users dataset 1", df_common_users_1_filtered['user_id'].nunique(), "unique users")
    df_common_users_2_filtered = df_common_users_2[df_common_users_2['user_id'].isin(user_2)]
    print(len(df_common_users_2_filtered), "entries in filtered common users dataset 2", df_common_users_2_filtered['user_id'].nunique(), "unique users")
    df_common_users = df_common_users_2[df_common_users_2['user_id'].isin(common_users)]
    print(len(df_common_users), "entries in common users dataset", df_common_users['user_id'].nunique(), "unique users")

    df_common_users_1_filtered.to_csv(os.path.join(os.path.dirname(file_path_common_users_1), 'filtered_'+os.path.basename(file_path_common_users_1)), index=False)
    df_common_users_2_filtered.to_csv(os.path.join(os.path.dirname(file_path_common_users_2), 'filtered_'+os.path.basename(file_path_common_users_2)), index=False)
    df_common_users.to_csv(os.path.join(os.path.dirname(file_path_common_users_2), 'common_users_'+os.path.basename(file_path_common_users_2)+os.path.basename(file_path_common_users_1)), index=False)
    

if __name__ == "__main__":
    # # Example usage of number_of_reviews function
    # print(number_of_reviews('Data/Original main datasets/Clothing_Shoes_and_Jewelry.jsonl.gz', 'user_id'))


    # # example usage of fetch_common_users_from_high_resource_domain function
    # print("=== Amazon Reviews Processing Script ===")
    
    # # Prepare DataFrame for LLM enrichment
    # high_resource = 'Data/Clothing_Shoes_and_Jewelry/'
    # low_resource = 'Data/All_Beauty/All_Beauty.jsonl'
    
    # print(f"Starting processing with:")
    # print(f"  High resource domain: {high_resource}")
    # print(f"  Low resource domain: {low_resource}")
    
    # fetch_common_users_from_high_resource_domain(high_resource, low_resource)
    
    # print("=== Script completed ===")

    # # example usage of fetch_common_users_from_low_resource_domain function
    # fetch_common_users_from_low_resource_domain('Data/Current Doamins/All_Beauty/All_Beauty_merged_whole_dataset.csv','Data/Current Doamins/Clothing_Shoes_and_Jewelry/Clothing_Shoes_and_Jewelry_merged_and_filtered_based_on_users.csv')
    
    
    divide_low_resource_dataset_for_llm_enrichment("Data/Current Doamins/All_Beauty/All_Beauty_Beauty_and_Personal_Care_common_users.csv", 'Data/Current Doamins/All_Beauty/All_Beauty_Clothing_Shoes_and_Jewelry_common_users.csv')