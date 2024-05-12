# Advanced MNIST+ classification with anomaly detection 

This project aims to solve the problem of classifying images from the MNIST+ advanced dataset, which contains not only familiar digits but also anomalous image "noise". A convolutional neural network capable of both digit recognition and anomaly detection is implemented. 

## High Performance Dependencies

- Python 3.12
- PyTorch 2.3.0
- OpenCV 4.9.0
- NumPy 1.26.4, Pandas 2.2.2
- Matplotlib 3.8.4

## Architecture 

Docker is used for ease of deployment, with `Dockerfile` and `docker-compose.yml`.

## Usage

For full development cycle, including anomaly generation, training and validation:

```
docker-compose up --build
```

Comprehensive pipeline:

1. Loads the MNIST source figures 
2. Generates realistic anomalous images using advanced algorithms
3. Combines the data into a new MNIST+ set
4. Trains a convolutional network for digit classification and anomaly detection
5. Validates the trained model
6. Visualizes the classification results
7. Saves the weights for deployment in production

Results are saved in `app/saves` directory outside Docker for later analysis.
