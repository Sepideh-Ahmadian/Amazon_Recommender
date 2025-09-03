import os
import csv
import pandas as pd
from openai import AzureOpenAI
from dotenv import load_dotenv
import time
from typing import List, Dict
import json
import re
import time
from typing import List, Dict

print("Starting Batch Review Processing...")

# Load environment variables from .env file
load_dotenv()

# Get variables from .env
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
DEPLOYMENT_NAME = os.getenv("CHAT_COMPLETION_NAME")

# Check that everything is loaded
if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_KEY or not DEPLOYMENT_NAME:
    raise ValueError("Missing one or more environment variables. Check your .env file.")

# Set up the AzureOpenAI client
client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version="2024-08-01-preview",
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

def read_reviews_from_csv(file_path: str, review_column: str = 'review', title_column: str = 'title') -> List[Dict]:
    """
    Read reviews and titles from a CSV file
    
    Args:
        file_path: Path to the CSV file
        review_column: Name of the column containing reviews
        title_column: Name of the column containing product titles
    
    Returns:
        List of dictionaries containing review and title data
    """
    try:
        df = pd.read_csv(file_path, nrows=50)  # Limit to first 100 rows for testing
        print(f"Loaded {len(df)} reviews from {file_path}")
        
        # Check if required columns exist
        missing_columns = []
        if review_column not in df.columns:
            missing_columns.append(review_column)
        if title_column not in df.columns:
            missing_columns.append(title_column)
        
        if missing_columns:
            print(f"Available columns: {list(df.columns)}")
            raise ValueError(f"Columns {missing_columns} not found in CSV file")
        
        # Convert to list of dictionaries
        reviews = df.to_dict('records')
        return reviews
    
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        raise

def analyze_review_batch(review_data_list: List[Dict], review_column: str = 'review', title_column: str = 'title') -> str:
    """
    Send a batch of reviews with product titles to the LLM for analysis
    
    Args:
        review_data_list: List of dictionaries containing review and title data
        review_column: Name of the column containing review text
        title_column: Name of the column containing product titles
    
    Returns:
        Analysis result string
    """
    # Create a formatted text with both titles and reviews
    reviews_with_titles = []
    for i, item in enumerate(review_data_list):
        title = item.get(title_column, "N/A")
        review = item.get(review_column, "N/A")
        reviews_with_titles.append(f"Product {i+1}: {title}\nReview {i+1}: {review}")
    
    reviews_text = "\n\n".join(reviews_with_titles)
    
    batch_prompt = f"""
    Analyze the following {len(review_data_list)} product reviews along with their product titles as instructed below:  
    You are an assistant who extracts product aspects from Amazon reviews of clothing, fashion, and shoes. 
    ### Instructions:
    - Identify review phrases that describe product features. 
    - For each phrase, output in this format: 
        "<exact phrase from review>" (aspects, sentiment) 
    - Sentiment should be **+1 for positive** and **-1 for negative**. 
    - For each extracted aspect, add a single-word adjective that best describes it. The adjective should be chosen based on the context and the expressed sentiment.
    - Ignore irrelevant information (seller, or unrelated experiences).
    - Return only the product number + list of extracted features, no additional text.
    - Ensure that all extracted features are relevant to the fashion domain.


    ### Example Input:
    "These sneakers look stylish and are very comfortable, but they run a little small and the material feels cheap."


    ### Example Output:
    Product Number: ["look stylish" (Fashionable Appearance, +1), "very comfortable" (Comfortable Fit, +1), "run a little small" (Tight Size, -1), "material feels cheap" (Inferior Material, -1)]

    ### Now extract features from the following reviews:
    {reviews_text}
    
    """

    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are an expert in analyzing customer reviews and product information, extracting key product aspects while considering both the product name and customer feedback."},
                {"role": "user", "content": batch_prompt}
            ],
            max_tokens=4000,
            temperature=0.1
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        print(f"Error in API call: {e}")
        return f"Error processing batch: {str(e)}"


    """
    Process reviews in batches and collect results
    
    Args:
        reviews: List of review dictionaries
        batch_size: Number of reviews per batch
        review_column: Name of the column containing review text
        title_column: Name of the column containing product titles
    
    Returns:
        List of results with original data plus individual analysis for each review
    """
    results = []
    total_batches = (len(reviews) + batch_size - 1) // batch_size
    
    for i in range(0, len(reviews), batch_size):
        batch_num = (i // batch_size) + 1
        print(f"Processing batch {batch_num}/{total_batches}...")
        
        # Get current batch
        batch = reviews[i:i + batch_size]
        
        # Filter out items that don't have required columns
        valid_batch = []
        for item in batch:
            if review_column in item and title_column in item:
                if item[review_column] and item[title_column]:  # Check for non-empty values
                    valid_batch.append(item)
        
        if not valid_batch:
            print(f"No valid reviews found in batch {batch_num}")
            continue
        
        print(f"Processing {len(valid_batch)} valid items in batch {batch_num}")
        
        # Analyze the batch with both reviews and titles
        # This returns a list of analysis results, one for each review in the batch
        # batch_analysis_results = analyze_review_batch(valid_batch, review_column, title_column)
        # print(f"Batch analysis results: {batch_analysis_results}")
        # save_results_json(batch_analysis_results, "LLM_batch_results_temp.json")
           

        batch_analysis_results= load_results_json("LLM_batch_results_temp.json")
        print(f"Batch analysis results: {batch_analysis_results}")
        # Ensure we have matching number of results and reviews
        if len(batch_analysis_results) != len(valid_batch):
            print(f"Warning: Number of analysis results ({len(batch_analysis_results)}) doesn't match number of reviews ({len(valid_batch)})")
            # Handle mismatch by padding with empty results or truncating
            while len(batch_analysis_results) < len(valid_batch):
                batch_analysis_results.append([])  # Add empty analysis
            batch_analysis_results = batch_analysis_results[:len(valid_batch)]  # Truncate if too many
        
        # Store results with original data and individual analysis
        for j, review_data in enumerate(valid_batch):
            result = review_data.copy()
            result['batch_number'] = batch_num
            result['review_index_in_batch'] = j + 1
            
            # Add the individual analysis result for this specific review
            result['analysis_result'] = batch_analysis_results[j]
            
            # Optional: Parse the analysis into structured format for easier access
            if batch_analysis_results[j]:
                result['parsed_aspects'] = parse_analysis_result(batch_analysis_results[j])
            else:
                result['parsed_aspects'] = []
            
            results.append(result)
        
        # Add delay to respect rate limits
        if batch_num < total_batches:
            print("Waiting 2 seconds before next batch...")
            time.sleep(2)
    
    return results
def save_results_json(results: List[Dict], filename: str):
            """Save results to JSON file"""
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {filename}")
def load_results_json(filename: str) -> List[Dict]:
    """Load results from JSON file"""
    with open(filename, 'r', encoding='utf-8') as f:
        results = json.load(f)
    print(f"Loaded {len(results)} results from {filename}")
    return results

def parse_LLM_result_string_to_dataframe(analysis_string):
    """
    Parse product analysis string into a DataFrame with product_number and aspects_dict columns
    """
    print("DEBUG: Input string:")
    print(repr(analysis_string))  # Use repr to see exact characters
    print()
    
    product_sections = analysis_string.strip().split('\n\n')
    print(f"DEBUG: Found {len(product_sections)} product sections")
    results = []
    
    for i, section in enumerate(product_sections):
        print(f"DEBUG: Parsing section {i+1}:")
        print(repr(section))
        
        section = section.strip()
        if not section:
            print("DEBUG: Empty section, skipping")
            continue
        
        product_match = re.match(r'Product\s+(\d+):', section)
        if not product_match:
            print("DEBUG: No product match found")
            continue
        
        product_number = int(product_match.group(1))
        print(f"DEBUG: Product number: {product_number}")
        
        bracket_match = re.search(r'\[(.*)\]', section, re.DOTALL)
        print(f"DEBUG: Bracket match found: {bracket_match is not None}")
        
        if not bracket_match:
            print("DEBUG: No bracket content found")
            continue
        
        bracket_content = bracket_match.group(1)
        print(f"DEBUG: Bracket content (first 100 chars): {repr(bracket_content[:100])}")
        
        # Updated regex pattern to handle optional spaces better
        item_pattern = r'"([^"]+)"\s*\(([^,]+),\s*([+-]?\d+)\)'
        matches = re.findall(item_pattern, bracket_content)
        print(f"DEBUG: Found {len(matches)} matches")
        
        if not matches:
            print("DEBUG: No matches found, trying alternative patterns...")
            # Try alternative patterns for debugging
            quotes_only = re.findall(r'"([^"]+)"', bracket_content)
            print(f"DEBUG: Found {len(quotes_only)} quoted strings")
            parens_only = re.findall(r'\(([^)]+)\)', bracket_content)
            print(f"DEBUG: Found {len(parens_only)} parenthetical expressions")
        
        aspects_list = []
        for text, category, sentiment in matches:
            aspects_list.append({
                'text': text.strip(),
                'category': category.strip(), 
                'sentiment': int(sentiment)
            })
        
        results.append({
            'product_number': product_number,
            'aspects_dict': aspects_list
        })
        
        print(f"DEBUG: Successfully parsed {len(aspects_list)} aspects for product {product_number}")
        print()
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    return df

def process_reviews_in_batches(reviews: List[Dict], batch_size: int = 10, review_column: str = 'review', title_column: str = 'title') -> List[Dict]:
    """
    Process reviews in batches and collect results
    
    Args:
        reviews: List of review dictionaries
        batch_size: Number of reviews per batch
        review_column: Name of the column containing review text
        title_column: Name of the column containing product titles
    
    Returns:
        List of results with original data plus individual analysis for each review
    """
    results = []
    total_batches = (len(reviews) + batch_size - 1) // batch_size
    
    for i in range(0, len(reviews), batch_size):
        batch_num = (i // batch_size) + 1
        print(f"Processing batch {batch_num}/{total_batches}...")
        
        # Get current batch
        batch = reviews[i:i + batch_size]
        
        # Filter out items that don't have required columns
        valid_batch = []
        for item in batch:
            if review_column in item and title_column in item:
                if item[review_column] and item[title_column]:  # Check for non-empty values
                    valid_batch.append(item)
        
        if not valid_batch:
            print(f"No valid reviews found in batch {batch_num}")
            continue
        
        print(f"Processing {len(valid_batch)} valid items in batch {batch_num}")
        
        # Analyze the batch with both reviews and titles
        # This returns a list of analysis results, one for each review in the batch
        batch_analysis_results = analyze_review_batch(valid_batch, review_column, title_column)
        # print(f"Batch analysis results: {batch_analysis_results}")
        # save_results_json(batch_analysis_results, "LLM_batch_results_temp.json")
        # batch_analysis_results= load_results_json("LLM_batch_results_temp.json")
        print(f"Batch analysis results: {batch_analysis_results}")
        # Parse the string results into DataFrame
        parsed_df = parse_LLM_result_string_to_dataframe(batch_analysis_results)
        print("Parsed DataFrame:")
        print(parsed_df)
        print(f"Batch analysis results type: {type(batch_analysis_results)}")
        
        # Convert DataFrame to dictionary for easier lookup
        parsed_dict = {}
        for _, row in parsed_df.iterrows():
            parsed_dict[row['product_number']] = row['aspects_dict']
        
        print(f"Parsed dictionary keys: {list(parsed_dict.keys())}")
        print(f"Number of valid batch items: {len(valid_batch)}")
        
        # Store results with original data and individual analysis
        for j, review_data in enumerate(valid_batch):
            result = review_data.copy()
            result['batch_number'] = batch_num
            result['review_index_in_batch'] = j + 1
            
            # Map product number (j+1) to parsed analysis
            product_num = j + 1
            if product_num in parsed_dict:
                result['analysis_result'] = parsed_dict[product_num]
            else:
                print(f"Warning: No analysis found for product {product_num} in batch {batch_num}")
                result['analysis_result'] = []
                
            
            results.append(result)
        
        # Add delay to respect rate limits
        if batch_num < total_batches:
            print("Waiting 2 seconds before next batch...")
            time.sleep(2)
    
    return results

def save_results_to_csv(results: List[Dict], output_file: str):
    """
    Save results to a CSV file
    
    Args:
        results: List of result dictionaries
        output_file: Path for output CSV file
    """
    try:
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
        print(f"Total records saved: {len(results)}")
        
        # Print sample of what was processed
        if len(results) > 0:
            print("\nSample of processed data:")
            sample = results[0]
            print(f"Product Title: {sample.get('title', 'N/A')}")
            print(f"Review: {sample.get('review', 'N/A')[:100]}...")
            print(f"Analysis: {sample.get('analysis_result', 'N/A')[:100]}...")
    
    except Exception as e:
        print(f"Error saving results: {e}")
        raise

def main():
    """Main function to orchestrate the batch processing"""
    
    # Configuration
    input_csv_file = "Data/Current Doamins/All_Beauty/merged.csv"  # Change this to your input CSV file path
    output_csv_file = "Data/LLM Output sampels/LLM_result_All_Beauty.csv"  # Output file path
    review_column_name = "review"  # Change this to match your CSV column name
    title_column_name = "product_title"   # Change this to match your CSV title column name
    batch_size = 10
    
    try:
        # Step 1: Read reviews and titles from CSV
        print("Step 1: Reading reviews and product titles from CSV...")
        reviews = read_reviews_from_csv(input_csv_file, review_column_name, title_column_name)
        print(f"Total reviews read: {len(reviews)}")
        if not reviews:
            print("No reviews found in the CSV file.")
            return
        
        # Print info about what we're processing
        sample_review = reviews[0] if reviews else {}
        print(f"Sample product title: {sample_review.get(title_column_name, 'N/A')}")
        print(f"Sample review: {sample_review.get(review_column_name, 'N/A')[:100]}...")
        
        # Step 2: Process reviews in batches
        print(f"Step 2: Processing {len(reviews)} reviews with titles in batches of {batch_size}...")
        results = process_reviews_in_batches(reviews, batch_size, review_column_name, title_column_name)
       
        
        # Step 3: Save results to CSV
        print("Step 3: Saving results to CSV...")
        save_results_to_csv(results, output_csv_file)
        
        print("########################################################")
        print("Batch processing completed successfully!")
        print(f"Input file: {input_csv_file}")
        print(f"Output file: {output_csv_file}")
        print(f"Total reviews processed: {len(reviews)}")
        print(f"Batch size: {batch_size}")
        print(f"Review column: {review_column_name}")
        print(f"Title column: {title_column_name}")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()