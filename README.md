# Movie Recommendation API

This is a Flask application that provides a simple API for movie recommendations based on user ratings.

## Getting Started

These instructions will help you run the application on your local machine for development and testing purposes.

### Prerequisites

- Python 3.x
- Flask

### Installing

1. Clone the repository or download the source code.
2. Navigate to the project directory.
3. Install the required dependencies by running:

```
pip install flask
```

### Running the Application

1. Open a terminal or command prompt.
2. Navigate to the project directory.
3. Run the Flask application with the following command:

```
python app.py
```

The application will start running on `http://localhost:5000`.

## Endpoints

### Test Endpoints

- `/test` (GET): This endpoint returns a simple message to check if the Flask app is running.
- `/model_check` (GET): This endpoint returns a JSON response indicating if the recommendation model is up and running.

### Recommendation Endpoint

- `/recommend` (POST): This endpoint accepts a JSON payload containing the user ID and a list of movie ratings. It returns a JSON response with a list of recommended movies.

## Testing with Sample JSON Payload

To test the `/recommend` endpoint with a sample JSON payload, follow these steps:

1. Open a tool like Postman or use `curl` in your terminal.
2. Set the HTTP method to `POST`.
3. Set the URL to `http://localhost:5000/recommend`.
4. Go to the "Body" tab (or use the `-d` flag for `curl`).
5. Select the "raw" option and choose "JSON" from the dropdown menu (or use the `-H "Content-Type: application/json"` flag for `curl`).
6. Enter the following JSON payload:

```json
{
    "userid": 5,
    "ratings": [
        [3, 4.0],
        [50, 1.0],
        [216, 3.5]
    ]
}
```

7. Click the "Send" button (or run the `curl` command).

You should receive a JSON response containing a list of recommended movies similar to the following:

```json
[
    {
        "movie_id": 12345,
        "movie_name": "The Godfather",
        "imdb_id": "tt0068646"
    },
    {
        "movie_id": 67890,
        "movie_name": "The Shawshank Redemption",
        "imdb_id": "tt0111161"
    },
    {
        "movie_id": 24680,
        "movie_name": "The Dark Knight",
        "imdb_id": "tt0468569"
    },
    {
        "movie_id": 13579,
        "movie_name": "Schindler's List",
        "imdb_id": "tt0108052"
    },
    {
        "movie_id": 98765,
        "movie_name": "Pulp Fiction",
        "imdb_id": "tt0110912"
    }
]
```

Note that the recommended movies are hardcoded in this example. In a real-world scenario, you would implement your recommendation algorithm or integrate with a recommendation system.