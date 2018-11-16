# Skin-Lesion-Classification

## Goal 
In this task, we complete two independent binary image classification tasks that involve three unique diagnoses of skin lesions (melanoma, nevus, and seborrheic keratosis). In the first binary classification task, we distinguish between (a) melanoma and (b) nevus and seborrheic keratosis. In the second binary classification task, we distinguish between (a) seborrheic keratosis and (b) nevus and melanoma.

## Definitions:

Melanoma – malignant skin tumor, derived from melanocytes (melanocytic)
Nevus – benign skin tumor, derived from melanocytes (melanocytic)
Seborrheic keratosis – benign skin tumor, derived from keratinocytes (non-melanocytic)

## Data:
Lesion classification data includes the original image, paired with a gold standard (definitive) diagnosis, referred to as "Ground Truth".

### Training Image Data

2000 images are provided as training data, including 374 "melanoma", 254 "seborrheic keratosis", and the remainder as benign nevi (1372). The training data is provided as a ZIP file, containing dermoscopic lesion images in JPEG format and a CSV file with some clinical metadata for each image.

All images are named using the scheme ISIC_<image_id>.jpg, where <image_id> is a 7-digit unique identifier. EXIF tags in the images have been removed; any remaining EXIF tags should not be relied upon to provide accurate metadata.

The CSV file contains three columns:

image_id, identifying the image that the row corresponds to
age_approximate, containing the age of the lesion patient, rounded to 5 year intervals, or "unknown"
sex, containing the sex of the lesion patient, or "unknown"
Ground Truth Data

The Training Ground Truth file is a single CSV (comma-separated value) file, containing 3 columns:

The first column of each row contains a string of the form ISIC_<image_id>, where <image_id> matches the corresponding Training Data image.
The second column of each row pertains to the first binary classification task (melanoma vs. nevus and seborrheic keratosis) and contains the value 0 or 1.
The number 1 = lesion is melanoma
The number 0 = lesion is nevus or seborrheic keratosis
The third column of each row pertains to the second classification task (seborrheic keratosis vs. melanoma and nevus) and contains the value 0 or 1.
The number 1 = lesion is seborrheic keratosis
The number 0 = lesion is melanoma or nevus
Malignancy diagnosis data were obtained from expert consensus and pathology report information. Participants are not strictly required to limit development to the training data, and are free to train their algorithm using external data sources. However, any other sources of data in system development must be properly cited in the abstract.

## Project Structure

### TFrecord_datasets
Current;y contains just the training dataset converted to TF_records. Melanoma_training.tfrecords uses float32 for processing the image while Melanoma_training_uint8.tfrecords uses uint8.

### Code
Contains the code for the project

#### TFREcordCreator
Contains code for converting raw data to TFrecord for better performance using tf.data API

#### Train
Currently, contains a naive Convolution Neural Network to perform the first classification.
