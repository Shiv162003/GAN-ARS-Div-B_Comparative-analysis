# Pro GANs Architecture for Celeb Face Image Generation

## Overview
This repository contains the implementation of the Progressive Growing of GANs (Generative Adversarial Networks) architecture, specifically tailored for generating high-resolution celebrity face images. The model is trained on a dataset of celebrity face images and is capable of generating realistic and high-quality images of faces.
![image](https://github.com/Shiv162003/GAN-ARS-Div-B_Comparative-analysis/assets/120489897/988a36af-5ae6-4a1d-b985-4009ab77d3cd)

## Features
- **Progressive Growing:** The architecture follows the progressive growing technique, where the generator and discriminator progressively grow in resolution during training. This helps in generating high-resolution images with finer details.
- **Celeb Face Dataset:** The model is trained on a dataset consisting of celebrity face images, ensuring that the generated images resemble realistic celebrity faces.
- **High-Quality Image Generation:** The trained model is capable of generating high-quality images of celebrity faces, capturing various facial features and expressions.

## Dependencies
- Python 3.x
- PyTorch
- Other dependencies as specified in the `requirements.txt` file

## Usage
1. **Clone the Repository:** Clone this repository to your local machine using:
2. **Install Dependencies:** Navigate to the cloned repository and install the required dependencies using:
3. **Download Pretrained Model (Optional):** If a pretrained model is available, download it and place it in the appropriate directory.
4. **Generate Images:** Use the provided scripts or Jupyter notebooks to generate images using the trained model. Ensure that the paths to the model weights and other necessary files are correctly set.

## Example Usage
```python
# Import necessary libraries
import torch
from pro_gans import ProGAN

# Load pretrained model
model = ProGAN()
model.load_state_dict(torch.load('path_to_pretrained_model.pth'))

# Generate images
generated_images = model.generate_images(num_images=10, noise_seed=42)

# Save generated images
for i, image in enumerate(generated_images):
 image.save(f'generated_image_{i}.png')
Acknowledgements
This implementation is based on the original paper:

T. Karras, T. Aila, S. Laine, J. Lehtinen. "Progressive Growing of GANs for Improved Quality, Stability, and Variation." arXiv:1710.10196, 2017
