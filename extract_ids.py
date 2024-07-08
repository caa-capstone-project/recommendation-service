import csv

def extract_and_store_movie_ids(csv_file_path, output_file_path):
    with open(csv_file_path, 'r') as file:
        reader = csv.reader(file)
        first_line = next(reader)  # Read the first line
    # Extract movie ids from the first line, ignoring the first column (userid)
    movie_ids = [int(col.split('_')[1]) for col in first_line[1:]]
    
    # Store movie_ids in a separate file
    with open(output_file_path, 'w') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(movie_ids)

# Example usage
csv_file_path = 'data.csv'
output_file_path = 'movie_ids.csv'
extract_and_store_movie_ids(csv_file_path, output_file_path)