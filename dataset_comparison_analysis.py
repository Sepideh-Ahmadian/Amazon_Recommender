import os
import pandas as pd
import gzip
import json
## how many reviews have a user_id?
## how many reviews have a parent_asin?
def read_zip_file(filename, selected_fields=None):

    if selected_fields is None:
        print("No selected fields provided")
        return 0

    data = []

    input_path = f'Data/Original datasets/{filename}'

    with gzip.open(input_path, 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                row = {field: obj.get(field, "") for field in selected_fields}
                data.append(row)
            except json.JSONDecodeError:
                continue  # skip malformed lines

    df = pd.DataFrame(data)

    return df

def common_users_in_pair_domains():
        # Step 1: Load and sort filenames
    filenames = sorted(os.listdir('Data/Original datasets'))
    filenames = filenames[1:]  # Skip the first file if needed

    # Step 2: Initialize empty DataFrame for results
    intersection_df = pd.DataFrame(index=filenames, columns=filenames)

    # Step 3: Loop through upper triangle only
    for i, first_file in enumerate(filenames):
        df_first = read_zip_file(first_file,['user_id'])
        print(df_first)
        df_first = set(df_first['user_id'].tolist())
        for j in range(i, len(filenames)):
            second_file = filenames[j]

            df_second = read_zip_file(second_file,['user_id'])
            df_second = set(df_second['user_id'].tolist())
            print(f"Comparing {first_file} and {second_file}:")
            print(f"Users in {first_file}: {len(df_first)}")
            print(f"Users in {second_file}: {len(df_second)}")

            intersection = len(df_first.intersection(df_second))
            print(f"Common users: {intersection}")

            # Save intersection count to both upper and lower triangle
            intersection_df.loc[first_file, second_file] = intersection
            intersection_df.loc[second_file, first_file] = intersection

            del df_second

        del df_first

        # Step 4: Save partial result after each row
        intersection_df.to_csv('Data/Intersection results/intersections_partial.csv')

    # Step 5: Final save
    intersection_df.to_csv('Data/Intersection results/intersections.csv')

def reviews_per_productOrUser_in_domain(domain_name,product_user):
    if product_user:
        product_user = 'user_id'
    else:
        product_user = 'parent_asin'
    df = read_zip_file(domain_name,[product_user, 'title', 'text'])
    print(df.columns)
    df['review'] = df['title'] + " " + df['text']
    

    # Drop duplicate reviews per product
    df = df.drop_duplicates()

    # Group by product
    result = df.groupby(product_user).agg(
        num_reviews=('review', 'count'),
        all_reviews=('review', lambda texts: ' '.join(texts))
    ).reset_index()

    # Save to CSV
    result.to_csv('Dataset Analysis/'+product_user+'_review_'+domain_name+'.csv', index=False, encoding='utf-8')




if __name__ == "__main__":
    # Gives number of common users in each pair of domains
    reviews_per_productOrUser_in_domain("All_Beauty.jsonl.gz",0)
 