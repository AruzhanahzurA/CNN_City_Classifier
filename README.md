# CNN Astana and Almaty Classifier

This project focuses on classifying images of Astana and Almaty using Convolutional Neural Networks (CNNs). The dataset used for this project contains images of both cities.

## Requirements

- Python 3.x
- pandas
- PyTorch
- TensorFlow
- scikit-learn

## Installation

1. Clone the repository:

```
git clone https://github.com/yourusername/cnn-astana-almaty-classifier.git
```

2. Install the required dependencies:

```
pip install pandas torch tensorflow scikit-learn
```

## Usage

1. Navigate to the project directory:

```
cd cnn-astana-almaty-classifier
```

2. Run the script to gather data:

```
python gather_data.py
```

This script will gather and preprocess images of Astana and Almaty for training the CNN classifier.

3. Develop and train the CNN classification model using the gathered data:

```
python train_cnn_classifier.py
```

This script will train the CNN model with Dropout, Batch Normalization, and max pooling layers for optimization.

4. Evaluate the trained model by obtaining the F1 score and confusion matrix:

```
python evaluate_model.py
```

This script will calculate the F1 score and generate the confusion matrix to assess the classification performance of the trained CNN model.

## Model Architecture

The CNN classification model consists of convolutional layers followed by fully connected layers. Dropout layers are incorporated to prevent overfitting by randomly dropping neurons during training. Batch Normalization layers ensure stable training by normalizing the activations. Max pooling layers reduce the spatial dimensions of the feature maps, aiding in computational efficiency.

## Results

After training and evaluating the CNN classifier, the F1 score and confusion matrix are obtained to measure the model's performance in distinguishing between images of Astana and Almaty.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
