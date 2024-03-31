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

This is the project done in Google Collab. To run it you will need to download the datesets to your Google Drive first. Make sure to note the file path where you've uploaded it.

## Model Architecture

The CNN classification model consists of convolutional layers followed by fully connected layers. Dropout layers are incorporated to prevent overfitting by randomly dropping neurons during training. Batch Normalization layers ensure stable training by normalizing the activations. Max pooling layers reduce the spatial dimensions of the feature maps, aiding in computational efficiency.

## Results

After training and evaluating the CNN classifier, the F1 score and confusion matrix are obtained to measure the model's performance in distinguishing between images of Astana and Almaty.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
