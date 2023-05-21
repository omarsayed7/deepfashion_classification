# Clothing Articles Classification

Repo containing the training of a classifier for clothing articles

# Installation

To use this project, first clone the repo on your device using the command below:

Clone the repo

```
https://github.com/omarsayed7/deepfashion_classification.git
```

Create new conda environment containing the dependencies in the [requirements file](requirements.yml) .

Using `mamba` is preferred it's faster and more stable. see [here](https://mamba.readthedocs.io/en/latest/installation.html#existing-conda-install) for installation steps.

```
#If using mamba
mamba env create -f requirements.yml

#If using conda
conda env create -f requirements.yml
```
Activate the created environment.

```
conda activate fashion_classification
```
If you are using only `pip` 

```
pip install -r requirements.txt
```
# Project Structure:
```
├── README.md
├── .gitignore
├── requirements.yml
├── requirements.txt
├── data
|   └── classification_data
|       │── fashion_data.csv
|       │── mapping.json
|       │── img
|       └── raw_annotations
├── models
|   |── efficientnet-b0_0.0001_True_64_categorical_crossentropy_0.5_10_V1
|   |    │── accuracy.png
|   |    │── args.json
|   |    │── best_model.pth
|   |    │── quantized_best_model.pth
|   |    │── final_model__2023-05-20.pth
|   |    │── loss.png
|   |    └── top_losses.csv
|   └── logging
|       └── experiments_logs.csv
├── src
|    ├── model
|    │   │── performance_measurement
|    |   |   │── calculate_flops.py
|    |   |   └── calculate_receptive_field.py
|    │   └── training
|    |   |   │── main.py
|    |   |   │── data_utils.py
|    |   |   │── dataset.py
|    |   |   │── evaluation_utils.py
|    |   |   │── models.py
|    |   |   └── training_utils.py
|    ├── data
         └── build_dataset.py
```
# How to use:
**1-Setup the directories to match the above file structure**
```
python main.py -script_mode setup
```
**2-Download the zip file of the images and place it inside `/data` directory**

**3-Unzip the images inside `/data/classification_data`**

**4-Download the `list_category_img.txt` from the raw dataset and place it inside `/data/classification_data/raw_annotations`**

**5- Build the dataset from raw annotations to a CSV file will be used in training and a mapping json file to map label_id to label_str**
```
python build_dataset.py -sampling_threshold 500 -maximum_num_images 1000

--sampling_threshold        Threshold for filtering out the minor classes
--maximum_num_images        max num of images per class
```
**Now you are ready for training!**

# Dataset Summary:
I used the [DeepFasion](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html) Dataset which is a large-scale clothes database for Clothing Category and Attribute Prediction, collected by the Multimedia Lab at the Chinese University of Hong Kong.

The classification benchmark was published in 2016. It evaluates the performance of the FashionNet Model in predicting 46 categories and 1000 clothes attributes. For the original paper please refer to [DeepFashion: Powering Robust Clothes Recognition and Retrieval with Rich Annotations, CVPR 2016](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Liu_DeepFashion_Powering_Robust_CVPR_2016_paper.pdf).

The DeepFashion Database contains several datasets. In this project, the Category and Attribute Prediction Benchmark was used. This dataset contains 289,222 diverse clothes images from 46 different categories.

Below is the histogram of the clothing 46 categories :
![picture](assets/distrib.png)

## Note:
I have filtered out some of the classes with minimum images (roughly smalled than 1000 images) and ended up with 22 class and 22000 total number of images and that is because the lack of hardware resources to train and conduct multiple experiments on the full dataset.

# Modeling Expermints:
This section gives the details of the results of my experiments with `VGG-16`, `ResNet-50`, `Efficientnet-B0`, and `Efficientnet-B7`. I used the mentioned backbones freezed and also finetuned its weights with the classification head.
## Overview of the backbones:
**`VGG-16`**

VGG is a commonly used neural network because it performs well, it was trained for weeks on a massive set of training data, it generalizes well to different use cases, It consists of 16 convolutional layers.
![picture](assets/vgg.png)

------------------------------------

**`ResNet-50`**

The ResNet-50 model used for this experiment consists of 48 convolutional layers, as well as a MaxPool and an Average Pool layer (48+1+1=50 layers). With the deeper network structure, better detection rates are achieved indeed than with the flatter network structures previously used.s.
![picture](assets/resnet.png)

------------------------------------

**`EfficientNet-B0 and EfficientNet-B7`**

EfficientNet is a mobile friendly pure convolutional model (ConvNet) that proposes a new scaling method that uniformly scales all dimensions of depth/width/resolution using a simple yet highly effective compound coefficient.
![picture](assets/efficientnet_overview.png)

With considerably fewer numbers of parameters, the family of models are efficient and also provide better results. 
The total number of layers in EfficientNet-B0 the total is 237 and in EfficientNet-B7 the total comes out to 813!! But don’t worry all these layers can be made from 5 modules shown below.

![picture](assets/efficientnet_arch.png)

------------------------------------

## Training Results:

Idealy you can run the following script in order to conduct a training experiment 

``` 
python main.py -script_mode train -epochs 20 -model_name resnet-50 -use_wandb True -freeze_backbone True -experiment_version 1

--experiment_version        Version of experiment.
--epochs                    Number of train epochs.
--learning_rate             Learning rate of the optimizer.
--lr_scheduler              Use learning rate scheduler or not.
--batch_size                Batch size.
--loss                      training loss function. 
--dropout                   dropout of the classification head, or put None if you do not need dropout.
--model_name                Backbone name ("vgg-16", "resnet-50", "efficientnet-b0", "efficientnet-b7").
--freeze_backbone           Either freezing the backbone layers or not.
--use_wandb                 Track the experiments with Weights and Biases.
--eval_mode                 Evaluation and exporting the classification report on any of a subset of data (train, valid, or test).
--calculate_top_losses      Either Calculating top losses of the training data or not.
--script_mode               train mode or setup the data and folder structure mode.
```

**Data Loadining** 

* Train data: `19491`, Validation data: `3174`, Test data: `3174`
* Image size `(256,256,3)`
* Channel Normalization `(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)`

**Training Hyperparameters**

  `Batch Size` = 32

  `Epochs` = 20

  `optimizer` = Adam

  `lr` = 0.001

  `dropout` = 0.5

  `loss` = categorical_crossentropy

**Training Modifications**
* **Saving Best model while training:** Measuring validation accuracy for each epoch and save the model from the best validation epoch
* **Learning Rate Scheduler:** To improve the convergence and stability of deep learning models
* **Gradient Scaling:** scaling our gradients by some factor, so they aren't flushed to zero. 
* **Garbage Collection:** To free up memory and cache
* **Track experiments using Weights and Biases:** W&B Python Library to track machine learning experiments with a few lines of code. and also we can review the results in an interactive dashboard

**Experiments Results**

~  | VGG16 | EfficientNet-B0 | EfficientNet-B7 | ResNet-50
--- | --- | --- | --- | ---
Total Param. |135.3 M | 4.3 M | 46.4 M | 24 M
Trainable Param. |135.3 M | 4.0 M | 6.9 M|
Val. Accuracy | **86.88%**| 84.28%|83.12%|
Val. Macro-F1 | **86.88%**| 84.28%|83.12%|
Estimated FLOPS | 0.07 s |**0.002 s** |0.03 s| 0.02 s
FLOPs | 20.23 G |**0.53G** | 6.93 G | 5.41 G
Estimcated MACCs | 40.46 G |**1.06 G** |13.86 G| 10.82 G
Overall receptive field | 212 × 212 |299 × 299 |299 × 299| 96 × 96

***Conculsion***

increasing batch size.
decreasing the dropout 
decreasing the lr rate 

# Model Selection:

# Future Work:

# Reference



