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
from Database import SQLiteManager
from datetime import datetime

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


def analyze_review_batch(batch_size: int) -> str:
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
    importer = SQLiteManager("Amazon_Review.db")
    reviews = importer.fetch_data_for_llm_enrichment("Clothing_Common_Users", "meta_Clothing_Shoes_and_Jewelry", batch_size= batch_size)
    rowid_list =[]
    reviews_text_list = []
    for review in reviews:
        rowid_list.append(review[0])
        reviews_text_list.append(str(review[0])+": "+review[1])

    reviews_text = "\n\n".join(reviews_text_list)
    print(f"DEBUG: Reviews text for batch (first 500 chars): {repr(reviews_text[:500])}")
    
    batch_prompt = f"""
        Analyze the following {len(reviews_text)} product reviews for aspect-based sentiment analysis: 

        -TASK: Extract exact phrases that describe product features, performance, or user experience. Use the product title provided after "##" in each review if more context is needed. Each review starts with an ID number — keep the same ID in the output.

        -OUTPUT FORMAT:
        For each review, return the following list:[ReviewID, ["<exact phrase>": (AspectLabel, sentiment_score), ...]]

                -Sentiment score: float between -1.0 (negative) and +1.0 (positive)
                -AspectLabel: one adjective + one noun summarizing the aspect (e.g., Comfortable Fit, Inferior Material)
                -Only extract phrases directly related to the product itself
                -If no product aspects are found, return an empty list for that review

        EXAMPLE:
        Input:
        1: These sneakers look stylish and are very comfortable, but they run a little small, and the material feels cheap. ## Nike Jordan sneakers

        Output:
        [1, ["look stylish": (Fashionable Appearance, +1), "very comfortable": (Comfortable Fit, +1), "run a little small": (Tight Size, -1), "material feels cheap": (Inferior Material, -1)]]

        REQUIREMENTS:
        -Process only English reviews
        -Extract only product-related aspects (ignore shipping, packaging, seller, etc.)
        -Return review ID + extracted features only in json format (no extra commentary)
        -Apply only to reviews in the clothing and fashion domain


        {reviews_text}

        """

    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are an expert in analyzing customer reviews and product information, extracting key product aspects while considering both the product name and customer feedback."},
                {"role": "user", "content": batch_prompt}
            ],
            max_completion_tokens =4000,
            temperature=0.1
        )
        print(f"DEBUG: LLM response (first 500 chars): {repr(response.choices[0].message.content[:500])}")
        save_results_json(response.choices[0].message.content, "Output/LLM_response "+str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))+".json")
        return response.choices[0].message.content, rowid_list
    
    except Exception as e:
        print(f"Error in API call: {e}")
        return f"Error processing batch: {str(e)}", rowid_list

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

def parse_llm_response(response_string):
    """
    Parse LLM response string that contains JSON formatted data with Python tuple syntax.
    Handles empty results gracefully.
    
    Returns:
        tuple: (parsed_results, success_flag)
            - parsed_results: list of tuples (review_id, aspects_str)
            - success_flag: boolean indicating if parsing was successful
    """
    print("Response String:", response_string[:200] + "..." if len(response_string) > 200 else response_string)
    
    # Handle empty or None responses
    if not response_string or response_string.strip() == "":
        print("Empty response string received.")
        return [], False
    
    try:
        # Clean the response - remove ```json and ``` markers if present
        cleaned_response = response_string.strip()
        if cleaned_response.startswith('```json'):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.startswith('```'):
            cleaned_response = cleaned_response[3:]
        if cleaned_response.endswith('```'):
            cleaned_response = cleaned_response[:-3]
        
        cleaned_response = cleaned_response.strip()
        
        # Fix Python tuple syntax to JSON array syntax
        # Convert (Category, score) to ["Category", score]
        cleaned_response = re.sub(r'\(([^,]+),\s*([^)]+)\)', r'["\1", \2]', cleaned_response)
        
        # Parse JSON
        json_data = json.loads(cleaned_response)
        
        parsed_results = []
        
        # Handle the JSON structure: [[id, [aspects]], [id, [aspects]], ...]
        for item in json_data:
            if isinstance(item, list) and len(item) >= 2:
                review_id = item[0]
                aspects_data = item[1]
                
                # Handle empty aspects list
                if not aspects_data or aspects_data == [] or aspects_data == "[]":
                    parsed_results.append((review_id, "[]"))
                    continue
                
                # Convert aspects to string format for compatibility
                if isinstance(aspects_data, list):
                    # Reconstruct the original format for your database
                    formatted_aspects = []
                    for aspect in aspects_data:
                        if isinstance(aspect, dict) and len(aspect) == 1:
                            # Extract phrase and (category, score)
                            phrase = list(aspect.keys())[0]
                            category_score = aspect[phrase]
                            if isinstance(category_score, list) and len(category_score) == 2:
                                formatted_aspects.append(f'"{phrase}": ({category_score[0]}, {category_score[1]})')
                    
                    aspects_str = "[" + ", ".join(formatted_aspects) + "]" if formatted_aspects else "[]"
                    parsed_results.append((review_id, aspects_str))
                else:
                    parsed_results.append((review_id, str(aspects_data)))
        
        print(f"Successfully parsed JSON format with {len(parsed_results)} items")
        return parsed_results, True
        
    except (json.JSONDecodeError, Exception) as e:
        print(f"JSON parsing failed: {e}")
        
        # Alternative approach: Extract data using regex patterns
        try:
            parsed_results = []
            
            # Find all review blocks: [number, [aspects]] including empty ones
            # Updated pattern to capture empty brackets
            pattern = r'\[(\d+),\s*(\[.*?\])\s*\]'
            matches = re.findall(pattern, response_string, re.DOTALL)
            
            for match in matches:
                review_id = int(match[0])
                aspects_text = match[1].strip()
                
                # Check if aspects are empty
                if aspects_text == "[]" or aspects_text.strip() == "[]":
                    parsed_results.append((review_id, "[]"))
                else:
                    # Clean up the aspects text and format it
                    # Remove outer brackets, clean, then re-add
                    inner_text = aspects_text[1:-1].strip()
                    formatted_aspects = f"[{inner_text}]" if inner_text else "[]"
                    parsed_results.append((review_id, formatted_aspects))
            
            if parsed_results:
                print(f"Successfully parsed using regex with {len(parsed_results)} items")
                print(f"print len parsed {len(parsed_results)}")
                if parsed_results:
                    print(f"Final parsed response example: {parsed_results[1]}")
                    
                    
                return parsed_results, True
            else:
                print("Regex parsing also failed - no matches found")
                return [], False
                
        except Exception as regex_error:
            print(f"Regex parsing failed: {regex_error}")
            return [], False

def parse_llm_response1(response_string):
    """
    Parse LLM response string that contains JSON formatted data with Python tuple syntax.
    """
    print("Response String:", response_string[:200] + "..." if len(response_string) > 200 else response_string)
    
    if response_string == "":
        print("Empty response string received.")
        return [], True
    
    try:
        # Clean the response - remove ```json and ``` markers if present
        cleaned_response = response_string.strip()
        if cleaned_response.startswith('```json'):
            cleaned_response = cleaned_response[7:]  # Remove ```json
        if cleaned_response.endswith('```'):
            cleaned_response = cleaned_response[:-3]  # Remove ```
        
        # Fix Python tuple syntax to JSON array syntax
        # Convert (Category, score) to ["Category", score]
        cleaned_response = re.sub(r'\(([^,]+),\s*([^)]+)\)', r'["\1", \2]', cleaned_response)
        
        # Parse JSON
        json_data = json.loads(cleaned_response.strip())
        
        parsed_results = []
        
        # Handle the JSON structure: [[id, [aspects]], [id, [aspects]], ...]
        for item in json_data:
            if isinstance(item, list) and len(item) >= 2:
                review_id = item[0]
                aspects_data = item[1]
                
                # Convert aspects to string format for compatibility
                if isinstance(aspects_data, list):
                    # Reconstruct the original format for your database
                    formatted_aspects = []
                    for aspect in aspects_data:
                        if isinstance(aspect, dict) and len(aspect) == 1:
                            # Extract phrase and (category, score)
                            phrase = list(aspect.keys())[0]
                            category_score = aspect[phrase]
                            if isinstance(category_score, list) and len(category_score) == 2:
                                formatted_aspects.append(f'"{phrase}" ({category_score[0]}, {category_score[1]})')
                    
                    aspects_str = "[" + ", ".join(formatted_aspects) + "]"
                    parsed_results.append((review_id, aspects_str))
                else:
                    parsed_results.append((review_id, str(aspects_data)))
        
        print(f"Successfully parsed JSON format with {len(parsed_results)} items")
        if len(parsed_results) ==0:
            return []
        else:
            return parsed_results
        
    except (json.JSONDecodeError, Exception) as e:
        print(f"JSON parsing failed: {e}")
        
        # Alternative approach: Extract data using regex patterns
        try:
            parsed_results = []
            
            # Find all review blocks: [number, [aspects]]
            pattern = r'\[(\d+),\s*\[(.*?)\]\]'
            matches = re.findall(pattern, cleaned_response, re.DOTALL)
            
            for match in matches:
                review_id = int(match[0])
                aspects_text = match[1].strip()
                
                # Clean up the aspects text and format it
                formatted_aspects = f"[{aspects_text}]"
                parsed_results.append((review_id, formatted_aspects))
            
            if parsed_results:
                print(f"Successfully parsed using regex with {len(parsed_results)} items")
                return parsed_results, True
            else:
                print("Regex parsing also failed")
                return [], False
                
        except Exception as regex_error:
            print(f"Regex parsing failed: {regex_error}")
            return [], False


def main():
    for item in range(34000):
        print("This is batch number ",str(item)," ",datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        llm_response, row_id = analyze_review_batch(10)
        #print("Row ids are as follows: ",row_id)
        #llm_response= load_results_json('Output/LLM_response 2025-09-29 10:28:55.json')
        #row_id = [2801920, 2801940,2801960, 2801980, 2802000, 2802020, 2802040, 1638440,  1638460, 1638480]
        if llm_response == []:
            print("No response from the LLM!", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            return
            
           
        else:
            #llm_response = load_results_json("result_results.json")
            parsed_reponse = parse_llm_response(llm_response)
            parsed_reponse=parsed_reponse[0]
            response_index = [t[0] for t in parsed_reponse]
            print("print len parsed",len(parsed_reponse))
            if set(response_index) != set(row_id):
                print("Mismatch between response indices and row IDs.")
                for index in  set(row_id) - set(response_index):
                    parsed_reponse.append((index, "No response from LLM!"))
                print("mismatch indexes are as follows: ",set(row_id) - set(response_index))
            parsed_reponse.sort(key=lambda x: x[0])  # Sort by rowid
            print(f"Final parsed response example: {parsed_reponse[0]}")
            print("lenght of final response: ", len(parsed_reponse))
            importer = SQLiteManager("Amazon_Review.db")
            importer.update_table_column("Clothing_Common_Users", "llm_enrichment", parsed_reponse)
        # time.sleep(2)    
       
       

if __name__ == "__main__":
    main()
    
