import os
import csv
import pandas as pd
from openai import AzureOpenAI
from dotenv import load_dotenv
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
        df = pd.read_csv(file_path)[:10]
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
    You are an expert in customer review analysis. 
    Your goal is to extract product aspects that customers either appreciate or criticize in digital music items (e.g., songs, albums, streaming content) sold on Amazon. 
    This helps identify customer tastes, priorities, and perceptions of the product.

    Instructions:
    Identify specific aspects mentioned in each review.
    Categorize each aspect into a clear feature domain.
    Assign sentiment polarity: +1 (positive), -1 (negative).
    Present results in the format: Aspect Mention:(CATEGORY, SENTIMENT)
    Example:
    "Great sound quality but slow shipping. Amazing lyrics."
    Output: [Great sound quality:(AUDIO_QUALITY,+1), Slow shipping:(DELIVERY_SPEED,-1), Amazing lyrics:(MUSICAL_CONTENT,+1)]

    {reviews_text}
    
    """
    # Suggested Categories:
    # AUDIO_QUALITY (recording, production, clarity)
    # MUSICAL_CONTENT (songs, lyrics, composition)
    # ARTIST_PERFORMANCE (vocals, how artist sounds)
    # PACKAGING, DELIVERY_SPEED, DELIVERY_QUALITY, SERVICE_QUALITY, AUTHENTICITY, VARIETY, PRICE.
    # Sentiment: +1 (positive), 0 (neutral), -1 (negative)
    # Format: [Feature:(CATEGORY,sentiment), ...]
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

def process_reviews_in_batches(reviews: List[Dict], batch_size: int = 10, review_column: str = 'review', title_column: str = 'title') -> List[Dict]:
    """
    Process reviews in batches and collect results
    
    Args:
        reviews: List of review dictionaries
        batch_size: Number of reviews per batch
        review_column: Name of the column containing review text
        title_column: Name of the column containing product titles
    
    Returns:
        List of results with original data plus analysis
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
        analysis_result = analyze_review_batch(valid_batch, review_column, title_column)
        
        # Store results with original data
        for j, review_data in enumerate(valid_batch):
            result = review_data.copy()
            result['batch_number'] = batch_num
            result['analysis_result'] = analysis_result
            result['review_index_in_batch'] = j + 1
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
    input_csv_file = "Data/Beauty_and_Personal_Care/merged_and_filtered_based_on_users.csv"  # Change this to your input CSV file path
    output_csv_file = "LLM_result_beauty_personal_care.csv"  # Output file path
    review_column_name = "review"  # Change this to match your CSV column name
    title_column_name = "product_title"   # Change this to match your CSV title column name
    batch_size = 10
    
    try:
        # Step 1: Read reviews and titles from CSV
        print("Step 1: Reading reviews and product titles from CSV...")
        reviews = read_reviews_from_csv(input_csv_file, review_column_name, title_column_name)
        
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