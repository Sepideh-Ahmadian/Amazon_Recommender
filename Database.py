import sqlite3
import csv
import os
from typing import List, Any, Optional
import json
import gzip
from tqdm import tqdm
import json
import ast
import pandas as pd

import time

class SQLiteManager:
    def __init__(self, db_path: str = "data.db"):
        """
        Initialize the CSV to SQLite converter.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.connection = None
        
    def connect(self):
        """Create connection to SQLite database."""
        try:
            self.connection = sqlite3.connect(self.db_path)
            print(f"Connected to database: {self.db_path}")
        except sqlite3.Error as e:
            print(f"Error connecting to database: {e}")
            raise
    
    def disconnect(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            print("Database connection closed")

    def jsonl_gz_to_sqlite(self, jsonl_gz_path: str, table_name: Optional[str] = None, 
                      batch_size: int = 1000, sample_size: int = 1000):
        """
        Import large compressed JSONL (.jsonl.gz) file into SQLite table with streaming processing.
        
        Args:
            jsonl_gz_path: Path to compressed JSONL file
            table_name: Name for the table (defaults to filename)
            batch_size: Number of rows to insert at once
            sample_size: Number of records to sample for schema detection
        """
        
        if not os.path.exists(jsonl_gz_path):
            raise FileNotFoundError(f"JSONL.gz file not found: {jsonl_gz_path}")
        
        if table_name is None:
            table_name = os.path.splitext(os.path.splitext(os.path.basename(jsonl_gz_path))[0])[0]
        
        print(f"Processing JSONL.gz file: {jsonl_gz_path}")
        print(f"Target table: {table_name}")
        
        # Phase 1: Count total lines and detect schema
        print("Analyzing file structure...")
        total_lines = 0
        all_keys = set()
        sample_count = 0
        
        with gzip.open(jsonl_gz_path, 'rt', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                total_lines += 1
                
                # Sample first N records for schema detection
                if sample_count < sample_size:
                    try:
                        record = json.loads(line.strip())
                        if isinstance(record, dict):
                            all_keys.update(record.keys())
                            sample_count += 1
                    except json.JSONDecodeError:
                        print(f"Warning: Invalid JSON at line {line_num}")
                        continue
        
        if not all_keys:
            raise ValueError("No valid JSON records found for schema detection")
        
        # Create headers and determine column types
        headers = sorted(list(all_keys))  # Sort for consistency
        clean_headers = []
        
        for header in headers:
            clean_header = str(header).strip().replace(' ', '_').replace('-', '_')
            clean_header = ''.join(c for c in clean_header if c.isalnum() or c == '_')
            if clean_header and not clean_header[0].isdigit():
                clean_headers.append(clean_header)
            else:
                clean_headers.append(f"col_{len(clean_headers)}")
        
        # Set column types
        column_types = []
        for header in clean_headers:
            header_lower = header.lower()
            if 'rating' in header_lower or 'price' in header_lower:
                column_types.append("REAL")
            else:
                column_types.append("TEXT")
        
        print(f"Found {len(headers)} columns in {total_lines:,} records")
        print(f"Schema: {clean_headers[:5]}...")
        
        # Phase 2: Create table
        self.create_table(table_name, clean_headers, column_types)
        
        # Phase 3: Stream and insert data
        cursor = self.connection.cursor()
        placeholders = ', '.join(['?' for _ in clean_headers])
        insert_sql = f"INSERT INTO {table_name} VALUES ({placeholders})"
        
        inserted_rows = 0
        skipped_rows = 0
        batch_data = []
        
        print("Importing data...")
        
        with gzip.open(jsonl_gz_path, 'rt', encoding='utf-8') as file:
            pbar = tqdm(total=total_lines, desc=f"Importing {table_name}", unit="records")
            
            for line_num, line in enumerate(file, 1):
                try:
                    record = json.loads(line.strip())
                    
                    if not isinstance(record, dict):
                        skipped_rows += 1
                        pbar.update(1)
                        continue
                    
                    # Build row in same order as headers
                    row = []
                    for header in headers:
                        value = record.get(header, '')
                        
                        # Handle nested objects/arrays
                        if isinstance(value, (dict, list)):
                            value = json.dumps(value)
                        elif value is None:
                            value = ''
                        else:
                            value = str(value)
                        
                        # Clean problematic characters
                        value = value.replace('\x00', '').replace('\x0b', '').replace('\x0c', '')
                        row.append(value)
                    
                    batch_data.append(row)
                    
                    # Insert batch when it reaches batch_size
                    if len(batch_data) >= batch_size:
                        cursor.executemany(insert_sql, batch_data)
                        self.connection.commit()
                        inserted_rows += len(batch_data)
                        pbar.set_postfix({
                            'Inserted': f'{inserted_rows:,}',
                            'Skipped': f'{skipped_rows:,}'
                        })
                        batch_data = []
                    
                    pbar.update(1)
                    
                except json.JSONDecodeError:
                    skipped_rows += 1
                    pbar.update(1)
                    continue
                except Exception as e:
                    print(f"\nError processing line {line_num}: {e}")
                    skipped_rows += 1
                    pbar.update(1)
                    continue
            
            # Insert remaining rows
            if batch_data:
                cursor.executemany(insert_sql, batch_data)
                self.connection.commit()
                inserted_rows += len(batch_data)
            
            pbar.close()
        
        print(f"\nImport completed:")
        print(f"- Successfully imported: {inserted_rows:,} records")
        print(f"- Skipped records: {skipped_rows:,}")
        print(f"- Total processed: {inserted_rows + skipped_rows:,}")
    
    def create_table(self, table_name: str, headers: List[str], column_types: List[str], 
                    drop_if_exists: bool = True):
        """
        Create table in SQLite database.
        
        Args:
            table_name: Name of the table to create
            headers: List of column names
            column_types: List of column types
            drop_if_exists: Whether to drop table if it already exists
        """
        if not self.connection:
            raise Exception("No database connection")
        
        cursor = self.connection.cursor()
        
        if drop_if_exists:
            cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        
        # Create table SQL
        columns_def = []
        for header, col_type in zip(headers, column_types):
            columns_def.append(f"{header} {col_type}")
        
        create_sql = f"CREATE TABLE {table_name} ({', '.join(columns_def)})"
        
        try:
            cursor.execute(create_sql)
            self.connection.commit()
            print(f"Table '{table_name}' created successfully")
            print(f"Schema: {create_sql}")
        except sqlite3.Error as e:
            print(f"Error creating table: {e}")
            raise 
   
    def select_command_executer(self, sql_query: str, params: tuple = None):
        """
        Execute a SELECT command and return the results.
        
        Args:
            sql_query: SELECT SQL command to execute
            params: Parameters for the SQL query (optional)
        
        Returns:
            List of tuples containing the query results
        """
        if not self.connection:
            self.connect()
        
        cursor = self.connection.cursor()
        
        try:
            if params:
                cursor.execute(sql_query, params)
            else:
                cursor.execute(sql_query)
            
            results = cursor.fetchall()
            return results
            
        except sqlite3.Error as e:
            print(f"SQL Error: {e}")
            raise
    
    def add_column_to_table(self, table_name: str,column_name: str = "LLM_enrichment"):
        """
        Add 'LLM_enrichment' column if it doesn't exist, with default value False.
        
        Args:
            table_name: Name of the table to add the column to
        """
        if not self.connection:
            self.connect()
        
        cursor = self.connection.cursor()
        
        try:
            # Check if column exists
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            column_exists = any(col[1] == 'LLM_enrichment' for col in columns)
            
            if column_exists:
                print(f"Column 'LLM_enrichment' already exists in table '{table_name}'. Skipping.")
                return
            
            # Add the column with default value False
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN LLM_enrichment INTEGER DEFAULT 0")
            
            # Update all existing rows to have False (0) value
            cursor.execute(f"UPDATE {table_name} SET LLM_enrichment = 0")
            
            self.connection.commit()
            print(f"Successfully added 'LLM_enrichment' column to table '{table_name}' with default value False")
            
        except sqlite3.Error as e:
            print(f"Error adding column: {e}")
            raise
        
    def list_tables(self):
        """
        List all tables in the database with row counts and column information.
        
        Returns:
            List of dictionaries with table info
        """
        if not self.connection:
            self.connect()
        
        cursor = self.connection.cursor()
        
        try:
            # Query to get all table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            print("All the tables in the database:", tables)
            table_info = []
            
            print(f"Found {len(tables)} tables in the database:")
            print("=" * 80)
            
            for i, (table_name,) in enumerate(tables, 1):
                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = cursor.fetchone()[0]
                
                # Get column information
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                column_names = [col[1] for col in columns]
                
                # Store table info
                table_data = {
                    'name': table_name,
                    'row_count': row_count,
                    'column_count': len(column_names),
                    'columns': column_names
                }
                table_info.append(table_data)
                
                # Display table info
                print(f"{i}. {table_name}")
                print(f"   Rows: {row_count:,}")
                print(f"   Columns: {len(column_names)} ({', '.join(column_names)})")
                print()
            
            print("=" * 80)
            return table_info
            
        except sqlite3.Error as e:
            print(f"Error listing tables: {e}")
            raise

    def fetch_data_for_llm_enrichment(self, main_table_name: str, meta_table_name:str, batch_size: int = 10):
        """
        Fetch specific rowid, title, text, meta.title from  tables with this filtering llm_enrichment=0.
        
        Args:
            table_names: Name of the table to fetch from
            batch_size: Number of rows to fetch at once
        
        Returns:
            List of tuples: [(rowid, col1_value, col2_value, ...), ...]
        """
        if not self.connection:
            self.connect()
        
        cursor = self.connection.cursor()
        
        try:
            # Build the fetch query in one line
            # fetched all the review of the users (good for lowresource domain)
            # fetch_sql = f"""
            #             SELECT t1.rowid, CONCAT(t1.title, " ", t1.text, " ## ", t2.title) 
            #             FROM {main_table_name} t1
            #             JOIN {meta_table_name} t2 ON t1.parent_asin = t2.parent_asin
            #             WHERE llm_enrichment = False AND t1.user_id IN (SELECT user_id from Common_Users_Reviews LIMIT 5000)
            #             LIMIT {batch_size}
                        # """
            
            fetch_sql = f"""
                         SELECT t1.rowid, CONCAT(t1.title, " ", t1.text, " ## ", t2.title) 
                         FROM {main_table_name} t1
                         JOIN {meta_table_name} t2 ON t1.parent_asin = t2.parent_asin
                         WHERE llm_enrichment = False AND t1.user_id IN (SELECT user_id from Common_Users_Reviews LIMIT 5000) AND MOD(t1.rowid,20) =0
                         LIMIT {batch_size}
                        """
                       
            # AND user_id in(Select user_id from Common_Users_Reviews LIMIT 1000)
            cursor.execute(fetch_sql)
            rows = cursor.fetchall()
            
            print(f"Fetched {len(rows)} rows from table '{main_table_name}'")
            
            return rows
            
        except sqlite3.Error as e:
            print(f"Error fetching data: {e}")
            raise
    
    def merge_tables(self, table1_name, table2_name, output_table_name):
        """
        Merge two tables and remove duplicates using UNION, then drop original tables.
        
        Args:
            table1_name: Name of first table
            table2_name: Name of second table  
            output_table_name: Name for the merged output table
        """
        if not self.connection:
            self.connect()
        
        cursor = self.connection.cursor()
        
        try:
            # Get row counts before merge
            cursor.execute(f"SELECT COUNT(*) FROM {table1_name}")
            count1 = cursor.fetchone()[0]
            
            cursor.execute(f"SELECT COUNT(*) FROM {table2_name}")
            count2 = cursor.fetchone()[0]
            
            # Create table and merge in one command using UNION
            cursor.execute(f"""
            CREATE TABLE {output_table_name} AS
            SELECT * FROM {table1_name}
            UNION
            SELECT * FROM {table2_name}
            """)
            
            # Get final row count
            cursor.execute(f"SELECT COUNT(*) FROM {output_table_name}")
            final_count = cursor.fetchone()[0]
            
            # Drop the original tables
            cursor.execute(f"DROP TABLE {table1_name}")
            cursor.execute(f"DROP TABLE {table2_name}")
            self.connection.commit()
            
            print(f"Tables merged successfully:")
            print(f"  {table1_name}: {count1:,} rows (dropped)")
            print(f"  {table2_name}: {count2:,} rows (dropped)")
            print(f"  {output_table_name}: {final_count:,} rows (duplicates removed)")
            
        except sqlite3.Error as e:
            print(f"Error merging tables: {e}")
            raise  

    def create_common_users_table(self, table1: str, table2: str, output_table1: str = None, output_table2: str = None):
        """
        Create new tables with records from users that exist in both input tables.
        
        Args:
            table1: First table name
            table2: Second table name  
            output_table1: Output table name for table1 records (defaults to table1_common_users)
            output_table2: Output table name for table2 records (defaults to table2_common_users)
        """
        if not self.connection:
            self.connect()
        
        cursor = self.connection.cursor()
        
        try:
            
            # Create table1 records for common users
            create_table1_query = f"""
                CREATE TABLE {output_table1} AS 
                SELECT t1.* 
                FROM {table1} t1 
                INNER JOIN (SELECT DISTINCT user_id FROM {table2}) t2 
                ON t1.user_id = t2.user_id
            """
            print(f"Creating table '{output_table1}'...")
            cursor.execute(f"DROP TABLE IF EXISTS {output_table1}")
            cursor.execute(create_table1_query)
            print(f"Created table '{output_table1}' with records from users common to both tables.")
            # Get count of records in first output table
            cursor.execute(f"SELECT COUNT(*) FROM {output_table1}")
            count1 = cursor.fetchone()[0]
            
            # Create table2 records for common users  
            create_table2_query = f"""
                CREATE TABLE {output_table2} AS 
                SELECT t2.* 
                FROM {table2} t2 
                INNER JOIN (SELECT DISTINCT user_id FROM {table1}) t1 
                ON t2.user_id = t1.user_id
            """
            cursor.execute(f"DROP TABLE IF EXISTS {output_table2}")
            cursor.execute(create_table2_query)
            
            # Get count of records in second output table
            cursor.execute(f"SELECT COUNT(*) FROM {output_table2}")
            count2 = cursor.fetchone()[0]
            
            self.connection.commit()
            
            print(f"Created tables:")
            print(f"- {output_table1}: {count1:,} records")
            print(f"- {output_table2}: {count2:,} records")
            
            return output_table1, output_table2
            
        except sqlite3.Error as e:
            print(f"Error creating common users tables: {e}")
            raise

    def update_table_column(self, table_name: str, update_column: str, row_data: list):
        """
        Update a specific column for multiple rows based on rowid.
        
        Args:
            table_name: Name of the table to update
            update_column: Name of the column to update
            row_data: List of tuples [(rowid, new_value), (rowid, new_value), ...]
        
        Returns:
            int: Number of rows successfully updated
        """
        if not self.connection:
            self.connect()
        
        cursor = self.connection.cursor()
        
        try:
            updated_count = 0
            
            print(f"Updating column '{update_column}' in table '{table_name}'...")
            
            # Update each row
            update_sql = f"UPDATE {table_name} SET {update_column} = ? WHERE rowid = ?"
            
            for row_id, new_value in row_data:
                try:
                    cursor.execute(update_sql, (new_value, row_id))
                    updated_count += 1
                    print(f"Row {row_id}: Updated {update_column} = {new_value}")
                    
                except Exception as e:
                    print(f"Row {row_id}: Error updating - {e}")
            
            self.connection.commit()
            print(f"Successfully updated {updated_count} rows")
            
            return updated_count
            
        except sqlite3.Error as e:
            print(f"Error updating table: {e}")
            raise   
    
    def save_query_to_csv(self, query, filename):
        """
        Execute query and save results to CSV file.
        
        Args:
            query: SQL query string
            filename: Output CSV filename
        """
        if not self.connection:
            self.connect()
        
        cursor = self.connection.cursor()
        
        try:
            # Execute query
            cursor.execute(query)
            results = cursor.fetchall()
            
            # Get column names
            column_names = [description[0] for description in cursor.description]
            
            # Write to CSV
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(column_names)
                writer.writerows(results)
            
            print(f"Query results saved to {filename}")
            print(f"Rows exported: {len(results)}")
            
        except Exception as e:
            print(f"Error saving query to CSV: {e}")
            raise
    
    def creat_common_users_review_table(self):
        """
        Extract users who are in the intersection of all three domains.
        count the number of their reviews in all the three domains and sort them from high to low based on the multiplication of their number of review in all domains.

        
        Args:
            input_table: Name of the input table
            output_table: Name of the output table to create
            min_reviews: Minimum number of reviews a user must have to be included
        """
        importer = SQLiteManager("Amazon_Review.db")
        importer.connect()
        cursor = importer.connection.cursor()
        print("Extracting users with high number of reviews...")
        try:
            # Create output table with users having at least min_reviews
            create_query = f"""
                CREATE TABLE Common_Users_Reviews AS
                SELECT 
                    t1.user_id,
                    t1.count1 as All_Beauty_Common_Users_count,
                    t2.count2 as Beauty_and_Personal_Care_Common_Users_count,
                    t3.count3 as Clothing_Common_Users_count,
                    (t1.count1 * t2.count2 * t3.count3) as multiplication_value
                FROM 
                    (SELECT user_id, COUNT(*) as count1 FROM All_Beauty_Common_Users GROUP BY user_id) t1
                INNER JOIN 
                    (SELECT user_id, COUNT(*) as count2 FROM Beauty_and_Personal_Care_Common_Users GROUP BY user_id) t2 ON t1.user_id = t2.user_id
                INNER JOIN 
                    (SELECT user_id, COUNT(*) as count3 FROM Clothing_Common_Users GROUP BY user_id) t3 ON t1.user_id = t3.user_id
                ORDER BY multiplication_value DESC; 
                """
            cursor.execute(create_query)
            
            importer.connection.commit()
            
            print(f"Created table is created.")
            
        except sqlite3.Error as e:
            print(f"Error extracting high-quality users: {e}")
            raise
        finally:
            importer.disconnect()

    def create_index(self, table_name: str, column_name: str, index_name: str = None):
        """
        Create an index on a specific column of a table.
        
        Args:
            table_name: Name of the table
            column_name: Name of the column to index
            index_name: Optional custom index name (auto-generated if not provided)
        """
        if not self.connection:
            self.connect()
        
        cursor = self.connection.cursor()
        
        try:
            # Generate index name if not provided
            if index_name is None:
                index_name = f"idx_{table_name}_{column_name}"
            
            # Check if index already exists
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='index' AND name=?
            """, (index_name,))
            
            if cursor.fetchone():
                print(f"Index '{index_name}' already exists on {table_name}.{column_name}")
                return
            
            # Create the index
            create_index_sql = f"CREATE INDEX {index_name} ON {table_name}({column_name})"
            
            print(f"Creating index '{index_name}' on {table_name}.{column_name}...")
            start_time = time.time()
            
            cursor.execute(create_index_sql)
            self.connection.commit()
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"Index '{index_name}' created successfully in {duration:.2f} seconds")
            
        except sqlite3.Error as e:
            print(f"Error creating index: {e}")
            raise
    
    def copy_table(self, source_table: str, new_table_name: str):
        """
        Create a copy of a table with a new name.
        
        Args:
            source_table: Name of the table to copy
            new_table_name: Name for the new copied table
        """
        if not self.connection:
            self.connect()
        
        cursor = self.connection.cursor()
        
        try:
            # Check if source table exists
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name=?
            """, (source_table,))
            
            if not cursor.fetchone():
                print(f"Error: Source table '{source_table}' does not exist")
                return
            
            # Check if new table name already exists
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name=?
            """, (new_table_name,))
            
            if cursor.fetchone():
                print(f"Error: Table '{new_table_name}' already exists")
                return
            
            # Get row count for progress tracking
            cursor.execute(f"SELECT COUNT(*) FROM {source_table}")
            total_rows = cursor.fetchone()[0]
            
            print(f"Copying table '{source_table}' to '{new_table_name}'...")
            print(f"Rows to copy: {total_rows:,}")
            
            start_time = time.time()
            
            # Create copy of table with all data
            copy_sql = f"CREATE TABLE {new_table_name} AS SELECT * FROM {source_table}"
            cursor.execute(copy_sql)
            
            self.connection.commit()
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Verify copy was successful
            cursor.execute(f"SELECT COUNT(*) FROM {new_table_name}")
            copied_rows = cursor.fetchone()[0]
            
            print(f"Table copied successfully in {duration:.2f} seconds")
            print(f"Original: {total_rows:,} rows | Copy: {copied_rows:,} rows")
            
        except sqlite3.Error as e:
            print(f"Error copying table: {e}")
            raise
    
    def reset_column_values(self, table_name: str, column_name: str, default_value=None):
        """
        Reset all values in a column to a default value.
        
        Args:
            table_name: Name of the table
            column_name: Name of the column to reset
            default_value: Value to set (None, 0, '', etc.)
        """
        if not self.connection:
            self.connect()
        
        cursor = self.connection.cursor()
        
        try:
            # Get row count for progress tracking
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            total_rows = cursor.fetchone()[0]
            
            print(f"Resetting column '{column_name}' in table '{table_name}'...")
            print(f"Rows to update: {total_rows:,}")
            
            start_time = time.time()
            
            # Update all values to default
            if default_value is None:
                update_sql = f"UPDATE {table_name} SET {column_name} = NULL"
            else:
                update_sql = f"UPDATE {table_name} SET {column_name} = ?"
                cursor.execute(update_sql, (default_value,))
            
            if default_value is None:
                cursor.execute(update_sql)
            
            self.connection.commit()
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"Column '{column_name}' reset successfully in {duration:.2f} seconds")
            print(f"All {total_rows:,} rows set to: {default_value}")
            
        except sqlite3.Error as e:
            print(f"Error resetting column: {e}")
            raise


def import_datasets_into_database():
    importer = SQLiteManager("Amazon_Review.db")
    
    importer.connect()

    # Import large compressed JSONL file
    importer.jsonl_gz_to_sqlite(
        "Data/Original main datasets/All_Beauty.jsonl.gz", 
        "All_Beauty",
        batch_size=5000,  # Larger batches for better performance
        sample_size=2000  # Sample more records for better schema detection
    )
    # Import large compressed JSONL file
    importer.jsonl_gz_to_sqlite(
        "Data/Original meta datasets/meta_All_Beauty.jsonl.gz", 
        "meta_All_Beauty",
        batch_size=5000,  # Larger batches for better performance
        sample_size=2000  # Sample more records for better schema detection
    )
    # Import large compressed JSONL file
    importer.jsonl_gz_to_sqlite(
        "Data/Original main datasets/Clothing_Shoes_and_Jewelry.jsonl.gz", 
        "Clothing_Shoes_and_Jewelry",
        batch_size=5000,  # Larger batches for better performance
        sample_size=2000  # Sample more records for better schema detection
    )
    # Import large compressed JSONL file
    importer.jsonl_gz_to_sqlite(
        "Data/Original meta datasets/meta_Clothing_Shoes_and_Jewelry.jsonl.gz", 
        "meta_Clothing_Shoes_and_Jewelry",
        batch_size=5000,  # Larger batches for better performance
        sample_size=2000  # Sample more records for better schema detection
    )
     # Import large compressed JSONL file
    importer.jsonl_gz_to_sqlite(
        "Data/Original main datasets/Beauty_and_Personal_Care.jsonl.gz", 
        "Beauty_and_Personal_Care",
        batch_size=5000,  # Larger batches for better performance
        sample_size=2000  # Sample more records for better schema detection
    )
    # Import large compressed JSONL file
    importer.jsonl_gz_to_sqlite(
        "Data/Original meta datasets/meta_Beauty_and_Personal_Care.jsonl.gz", 
        "meta_Beauty_and_Personal_Care",
        batch_size=5000,  # Larger batches for better performance
        sample_size=2000  # Sample more records for better schema detection
    )
    importer.disconnect()


if __name__ == "__main__":
    importer = SQLiteManager("Amazon_Review.db")
    
    ## The following function is repossible for importing datasets into the database.
    ## Uncomment and run it only once.
    # import_datasets_into_database()
    
    ## The following code make the common users tables. The result of these two calls are 4 tables.
    ## Only run it once.
    # importer.create_common_users_table(
    #     "All_Beauty", 
    #     "Clothing_Shoes_and_Jewelry",
    #     "All_Beauty_Cloth_Common_Users",
    #     "Clothing_Common_Users"
    # )
    # importer.create_common_users_table(
    #     "All_Beauty", 
    #     "Beauty_and_Personal_Care",
    #     "All_Beauty_Personal_Care_Common_Users",
    #     "Beauty_and_Personal_Care_Common_Users"
    # )
    
    # The result of the two previous function calls are 4 tables. There are two All_Beauty tables.
    # The following code merges these two tables into one and removes duplicates.
    # Only run it once.
    # importer.merge_tables('All_Beauty_Personal_Care_Common_Users','All_Beauty_Cloth_Common_Users','All_Beauty_Common_Users')
    
    ## Add the LLM_enrichment column to your table
    ## Just run it once for each table.
    # importer.add_column_to_table("Beauty_and_Personal_Care_Common_Users","llM_enrichment")
    # importer.add_column_to_table("Clothing_Common_Users","llM_enrichment")
    # importer.add_column_to_table("All_Beauty_Common_Users","llM_enrichment")

    # ## Save the results with GPT-5 mini model
    # importer.copy_table("All_Beauty_Common_Users", "All_Beauty_Common_Users_gpt5mini")
    # Reset the LLM_enrichment column to False in the new table
    # importer.reset_column_values("All_Beauty_Common_Users", "llm_enrichment", 0)

    # #make indexes on user_id columns for faster joins
    # importer.create_index("All_Beauty_Common_Users", "user_id")
    # importer.create_index("Beauty_and_Personal_Care_Common_Users", "user_id")
    # importer.create_index("Clothing_Common_Users", "user_id")

    # Extarct users with high number of reviews from common users table
    #importer.creat_common_users_review_table()

    ################### Playground test
    #This is an example of fetching data from a table         
    # result= importer.select_command_executer("SELECT count(rowid) from Clothing_Common_Users WHERE user_id in (SELECT user_id FROM Common_Users_Reviews LIMIT 1500)")
    # print(result)
    # result= importer.select_command_executer("SELECT user_id, parent_asin, llm_enrichment, rating from All_Beauty_Common_Users WHERE llm_enrichment<>0 LIMIT 5")
    # print(result)
    # print("&&&&&&&&&&&&&&&")
    result= importer.select_command_executer("""SELECT Count(DISTINCT(user_id)) 
                        FROM Clothing_Common_Users
                                             WHERE llm_enrichment<>False """)
    for item in result:
        print(item)
    
    # result= importer.select_command_executer("""SELECT * 
    #                     FROM All_Beauty_Common_Users_KG_Triplets
    #                                         LIMIT 5""")
    # for item in result:
    #     print(item)
    # result = importer.select_command_executer("SELECT count(*) from All_Beauty_Common_Users where llm_enrichment<>False group by user_id")
    # for item in result:
    #     print(item[0])
    #     print(item[1])
    # result=importer.fetch_data_for_llm_enrichment("All_Beauty_Common_Users", "meta_All_Beauty",10)
    # print(result)
    # # List all tables
    tables = importer.list_tables()

    # # Use the returned list
    print(f"Total tables: {len(tables)}")
    # for table in tables:
    #     print(f"Table: {table}")
    # # Save a query result to CSV
    # importer.save_query_to_csv("SELECT rowid, * FROM All_Beauty_Common_Users WHERE rowid>400 AND rowid<600", "high_ratings.csv")
    # print(result)
    # 'Beauty_and_Personal_Care', Clothing_Shoes_and_Jewelry, All_Beauty
    