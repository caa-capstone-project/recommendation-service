from flask import Flask, request, jsonify

app = Flask(__name__)

# Test endpoint to check if the app is running
@app.route('/test', methods=['GET'])
def test():
    return 'Flask app is running!'

# Test endpoint to check if the model is up
@app.route('/model_check', methods=['GET'])
def model_check():
    return jsonify({'status': 'Model is up and running'})

# Endpoint to handle the recommendation
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    user_id = data['userid']
    ratings = data['ratings']

    recommendations = [
    {
        'movie_id': 12345,
        'movie_name': 'The Godfather',
        'imdb_id': 'tt0068646'
    },
    {
        'movie_id': 67890,
        'movie_name': 'The Shawshank Redemption',
        'imdb_id': 'tt0111161'
    },
    {
        'movie_id': 24680,
        'movie_name': 'The Dark Knight',
        'imdb_id': 'tt0468569'
    },
    {
        'movie_id': 13579,
        'movie_name': 'Schindler\'s List',
        'imdb_id': 'tt0108052'
    },
    {
        'movie_id': 98765,
        'movie_name': 'Pulp Fiction',
        'imdb_id': 'tt0110912'
    }]

    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)