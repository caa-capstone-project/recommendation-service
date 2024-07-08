import pandas as pd
import csv
from collections import defaultdict

def process_csv_in_chunks(input_filepath, output_filepath, chunk_size=1000):
    user_ratings = defaultdict(dict)  # Use defaultdict with nested dicts

    # Read the input CSV in chunks using pandas
    chunks = pd.read_csv(input_filepath, chunksize=chunk_size)

    # Open the output file once and write header
    with open(output_filepath, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        
        # Prepare the header based on all unique movieIds in the dataset
        all_movie_ids = set()
        for chunk in chunks:
            all_movie_ids.update(chunk['movieId'].unique())
        all_movie_ids = sorted(all_movie_ids)
        
        header = ['userid'] + [f'movie_{movie_id}' for movie_id in all_movie_ids]
        writer.writerow(header)  # Write header
        
        # Rewind the input CSV to process it again
        chunks = pd.read_csv(input_filepath, chunksize=chunk_size)
        
        for chunk in chunks:
            # Process each chunk
            for index, row in chunk.iterrows():
                user_ratings[int(row['userId'])][int(row['movieId'])] = row['rating']
            
            # Write the processed data to the output file in chunks
            for userid, ratings in user_ratings.items():
                # Create a row with NaNs for movies not rated by the user
                row_data = [userid] + [ratings.get(movie_id, -1) for movie_id in all_movie_ids]
                writer.writerow(row_data)
            
            # Clear the user_ratings dictionary to free memory
            user_ratings.clear()

# Example usage
input_filepath = 'data/ratings.csv'
output_filepath = 'data_output.csv'
process_csv_in_chunks(input_filepath, output_filepath, chunk_size=1000)
