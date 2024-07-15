import csv
import torch
import model
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.Recommender()
model.load_state_dict(torch.load("model.pth"))
model.to(device)

def inference(input):
    with torch.no_grad():  # Disable gradient computation for inference
        output = model(input)
    return output

# Step 1: Load movie IDs from the stored file
def load_movie_ids(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        movie_ids = next(reader)  # Read the first line
    # Convert movie ids to integers
    movie_ids = [int(movie_id) for movie_id in movie_ids]
    return movie_ids

# Step 2: Create a dictionary to map movie_id to its index in the list
def create_movie_id_to_index_map(movie_ids):
    movie_id_to_index = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}
    return movie_id_to_index

# Step 3: Create a list initialized with -1 for all movie_ids
def create_ratings_list(movie_id_to_index, ratings):
    ratings_list = [-1] * len(movie_id_to_index)  # Create a list with all elements set to -1
    for movie_id, rating in ratings:
        if movie_id in movie_id_to_index:
            ratings_list[movie_id_to_index[movie_id]] = rating
    ratings_tensor = torch.tensor(ratings_list, dtype=torch.float32)
    ratings_tensor = ratings_tensor.to(device)  # Move the tensor to the GPU
    return ratings_tensor

def generate_random_ratings(n):
    # Initialize the tensor with -1
    ratings = torch.full((9724,), -1, dtype=torch.float32)
    
    # Possible ratings between 0.5 and 5.0 with 0.5 increments
    possible_ratings = [i * 0.5 for i in range(1, 11)]
    
    # Randomly select n unique indices for placing the ratings
    indices = random.sample(range(9724), n)
    
    for idx in indices:
        ratings[idx] = random.choice(possible_ratings)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ratings = ratings.to(device)
    return ratings


# Example usage
movie_ids_file_path = 'movie_ids.csv'
movie_ids = load_movie_ids(movie_ids_file_path)
movie_id_to_index = create_movie_id_to_index_map(movie_ids)

# Example input ratings
ratings_input = [
    [561, 4.0],
    [6548, 4.0],
    [216, 3.5],
    [4653,1.0]
]

# ratings = create_ratings_list(movie_id_to_index, ratings_input)

num_items = random.randint(100, 500)  # Generate a random number of items between 1 and 100
ratings = generate_random_ratings(num_items)
print(ratings)

predictions = inference(ratings)
no_of_recommendations = 5
top_values, top_indices = torch.topk(predictions, no_of_recommendations, largest=True, sorted=True)

print(predictions)
print(f"Top {no_of_recommendations} values: {top_values}")
print(f"Indices of top {no_of_recommendations} values: {top_indices}")
