import pandas as pd
import numpy as np
from tqdm import tqdm

# Function to generate new data items for each user
def augment_data(row):
    augmented_data = []
    valid_indices = [i for i, val in enumerate(row) if val != -1]
    
    for num_ratings in range(1, len(valid_indices) + 1):
        new_row = np.full(len(row), -1.0)
        for idx in valid_indices[:num_ratings]:
            new_row[idx] = row[idx]
        augmented_data.append(new_row)
    
    return augmented_data

# Path to the original and augmented CSV files
input_csv = 'data.csv'
output_csv = 'augmented_data.csv'

# Process the data in chunks
chunksize = 5  # Adjust this based on your memory constraints

# Get the total number of lines in the input CSV file for progress tracking
total_lines = sum(1 for _ in open(input_csv)) - 1  # Subtract 1 for the header

with pd.read_csv(input_csv, chunksize=chunksize) as reader:
    # Open the output CSV file
    with open(output_csv, 'w') as f:
        # Initialize the tqdm progress bar
        with tqdm(total=total_lines, unit='lines') as pbar:
            # Process each chunk
            for chunk in reader:
                all_augmented_data = []
                # Process each row (user) in the chunk
                for _, row in chunk.iterrows():
                    row = row.astype(float)
                    user_data = augment_data(row)
                    all_augmented_data.extend(user_data)
                
                # Convert the list to a DataFrame
                augmented_df = pd.DataFrame(all_augmented_data, columns=chunk.columns)
                
                # Write the DataFrame to the CSV file
                augmented_df.to_csv(f, index=False, header=f.tell()==0, lineterminator='\n')
                
                # Update the progress bar
                pbar.update(len(chunk))

print("Data augmentation completed and saved to augmented_data.csv")
