import csv
import torch
import model
from flask import Flask, request, jsonify

app = Flask(__name__)

NO_OF_RECOMMENDATIONS = 10
MODEL_STATUS = False
recommender = None
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    global recommender, MODEL_STATUS
    try:
        recommender = model.Recommender()
        # recommender.to(device)
        recommender.load_state_dict(torch.load("model.pth",  map_location=torch.device('cpu')))
        recommender.eval()  # Set model to evaluation mode
        MODEL_STATUS = True
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        MODEL_STATUS = False

def load_movie_ids(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        movie_ids = next(reader)  # Read the first line
    # Convert movie ids to integers
    movie_ids = [int(movie_id) for movie_id in movie_ids]
    return movie_ids

def create_movie_id_to_index_map(movie_ids):
    movie_id_to_index = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}
    return movie_id_to_index

def create_ratings_list(movie_id_to_index, ratings):
    ratings_list = [-1] * len(movie_id_to_index)  # Create a list with all elements set to -1
    for movie_id, rating in ratings:
        if movie_id in movie_id_to_index:
            ratings_list[movie_id_to_index[movie_id]] = rating
    ratings_tensor = torch.tensor(ratings_list, dtype=torch.float32)
    # ratings_tensor = ratings_tensor.to(device)  # Move the tensor to the GPU
    return ratings_tensor

movie_ids_file_path = 'movie_ids.csv'
movie_ids = load_movie_ids(movie_ids_file_path)
movie_id_to_index = create_movie_id_to_index_map(movie_ids)
load_model()

# Test endpoint to check if the app is running
@app.route('/test', methods=['GET'])
def test():
    return 'Flask app is running!'

# Test endpoint to check if the model is up
@app.route('/health', methods=['GET'])
def health_check():
    if MODEL_STATUS:
        return jsonify({"status": "healthy", "message": "Model is loaded and ready"}), 200
    else:
        return jsonify({"status": "unhealthy", "message": "Model is not loaded"}), 500

# Endpoint to handle the recommendation
@app.route('/recommend', methods=['POST'])
def recommend():
    global movie_id_to_index, NO_OF_RECOMMENDATIONS
    data = request.get_json()
    ratings = data['ratings']

    ratings = create_ratings_list(movie_id_to_index, ratings)

    with torch.no_grad():  # Disable gradient computation for inference
        predictions = recommender(ratings)

    _, top_indices = torch.topk(predictions, NO_OF_RECOMMENDATIONS, largest=True, sorted=True)

    movie_ids_list = top_indices.tolist()
    return jsonify({"recommendations": movie_ids_list})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4201, debug=True)