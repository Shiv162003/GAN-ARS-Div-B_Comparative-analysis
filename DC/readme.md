# Photorealistic Face Generator

This repository contains code for training a photorealistic face generator using Generative Adversarial Networks (GANs). The code is written in Python using the Keras library with TensorFlow backend.
![image](https://github.com/Shiv162003/GAN-ARS-Div-B_Comparative-analysis/assets/120489897/c99614b6-dfd3-433c-9a39-a1af6efa1700)
# Architecture
## Prerequisites
- Python 3.x
- Keras
- TensorFlow
- NumPy
- Matplotlib
- PIL (Python Imaging Library)

## Getting Started
1. Clone this repository to your local machine.
2. Ensure you have all the prerequisites installed.
3. Download the [Celebrity Faces Dataset](https://www.kaggle.com/dansbecker/5-celebrity-faces-dataset) and place it in the `Celebrity_Faces_Dataset` directory.
4. Open the terminal or command prompt and navigate to the directory containing the cloned repository.
5. Run the provided code using `python filename.py` (replace `filename.py` with the name of the file containing the code).

## Code Structure
- `load_and_preprocess_real_images`: Function to open, crop, and resize images from the dataset.
- `build_discriminator`: Function to build the discriminator model.
- `build_generator`: Function to build the generator model.
- `build_gan`: Function to build the GAN (Generative Adversarial Network) model.
- `generate_real_samples`: Function to generate real image samples from the dataset.
- `generate_noise_samples`: Function to generate random noise samples.
- `generate_fake_samples`: Function to generate fake image samples using the generator model.
- `generate_images`: Function to generate images from the generator model for a given epoch.
- `display_saved_images`: Function to display saved generated images after training.
- `plot_generated_images`: Function to plot and visualize generated images from the generator model for a given epoch.
- `train`: Function to train the generator and discriminator models.

## Usage
1. Adjust the hyperparameters such as `num_epochs`, `batch_size`, and `noise_dimension` according to your requirements.
2. Run the `train` function to start training the model. Optionally, set the `verbose` parameter to control the verbosity of the training process.

## Output
- The code generates photorealistic face images as output, demonstrating the progress of the generator during training.
- Optionally, you can save the trained generator model using the provided code snippet (commented out by default due to output file size constraints).
