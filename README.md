## Visualizing Bounding Boxes and Predictions

The ground-truth bounding boxes and predictive bounding boxes should be visualized, including their locations, categories, Intersection over Union (IoU) scores, and probabilities. An example image with locations and IoUs is shown below. In our specific task, category and probability information should also be included, which is not present in the example image:

![example_file](example_file.png)


## Analysing Classification Errors

To analyze classification errors, various tools can be utilized. For example, a confusion matrix can be employed to determine whether misclassifications primarily occur between specific classes, such as day1 and day3.

## Distribution of Categorical Predictions

The distribution of categorical predictions should be also analyzed. For instance, given that the ground truth label is day3-(0, 0, 1, 0, 0), the incorrect predictions should be examined to determine whether they are evenly distributed-(0.25, 0.2, 0.15, 0.2, 0.2) or highly centralized-(0.9, 0, 0.1, 0, 0, 0).

## Laser Localization Model

This is a PyTorch-based deep learning model for classifying laser images as either green or red based on the color of the laser. The model takes in laser images and predicts the corresponding label.

### Some Important Code Files

1. `test.py`
   This is the main Python script that contains the model architecture and training/evaluation pipelines. It imports the dataset classes from the other two files and uses them to evaluate the model.

2. `laser_data_cropped_day.py`
   This file contains the `LaserDayDataset` class, which is a PyTorch Dataset subclass used for loading and preprocessing the laser images for the daytime dataset. It defines the `load_annotations` function that reads the image filenames and corresponding labels from a text file and creates a list of data information dictionaries.

3. `laser_dataset.py`
   This file contains the `LaserDataset` class, which is a PyTorch Dataset subclass used for loading and preprocessing the laser images for the binary classification dataset. It defines the `load_annotations` function that reads the image filenames and corresponding labels from a text file and creates a list of data information dictionaries.

### Dataset

The laser classification model uses two datasets:

1. The binary classification dataset, which consists of laser images labeled as either green or red.
2. The daytime dataset, which consists of laser images captured during the day and labeled as one of five classes (day1 to day5).

### Training

The model is trained on the binary classification dataset and evaluated on the validation set of the daytime dataset. The training pipeline includes loading the images from file, preprocessing, augmentation, normalization, and converting to PyTorch tensors. The model is trained using the Adam optimizer and the cross-entropy loss function.

### Evaluation

The model is evaluated using the accuracy metric on the validation set of the daytime dataset. The evaluation pipeline includes loading the images from file, preprocessing, normalization, and converting to PyTorch tensors. The accuracy metric is used to evaluate the performance of the model.

## Laser Classification Model

Based on TensorFlow ... TBD

