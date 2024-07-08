from dataloader import *
import hyperparameters
import model
import train

if __name__ == "__main__":
    dataset = ChunkedRatingsDataset("augmented_data.csv", "data.csv", hyperparameters.CHUNK_SIZE)
    data_loader = DataLoader(dataset, batch_size=hyperparameters.BATCH_SIZE)
    model = model.Recommender()
    train.train_model(model, data_loader, num_epochs=hyperparameters.EPOCHS, learning_rate=hyperparameters.LEARNING_RATE)
    train.save_model(model)