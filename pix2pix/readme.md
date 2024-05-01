# Pix2Pix README

This repository contains an implementation of the Pix2Pix model for image-to-image translation using Keras with TensorFlow backend.

## Overview
![image](https://github.com/Shiv162003/GAN-ARS-Div-B_Comparative-analysis/assets/120489897/d0013690-d282-4e28-89de-1319490eb569)


Pix2Pix is a conditional adversarial network proposed for image-to-image translation tasks, which takes an input image and generates a corresponding output image. In this implementation, the Pix2Pix model is trained to translate images from one domain to another. Specifically, it learns to translate facade images to real building images.

## Dependencies

- Python 3.x
- TensorFlow
- Keras
- scipy
- matplotlib
- numpy

## Usage

1. Clone the repository:

    ```
    git clone https://github.com/your-username/pix2pix.git
    cd pix2pix
    ```

2. Install the dependencies:

    ```
    pip install -r requirements.txt
    ```

3. Run the training script:

    ```
    python pix2pix.py
    ```

## File Structure

- `pix2pix.py`: Main script containing the Pix2Pix model definition and training logic.
- `data_loader.py`: Data loader utility for loading and preprocessing the dataset.
- `images/`: Directory to store generated images during training.
- `datasets/`: Directory to store input datasets.

## Training

The Pix2Pix model is trained using the `train` method defined in `pix2pix.py`. You can adjust the training parameters such as the number of epochs, batch size, and sample interval according to your requirements.

## Sample Output

During training, sample output images are saved in the `images/` directory at specified intervals. These images include input images, generated images, and ground truth images for comparison.

## Credits

This implementation is based on the Pix2Pix paper by Phillip Isola et al. and the official Pix2Pix implementation provided by the TensorFlow team.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
