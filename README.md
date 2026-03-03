# Neural Style Transfer Model

 *COMPANY*: CODTECH IT SOLUTIONS
 
 *NAME*: MOHAMMED QAISUDDIN

 *INTERN ID*: CTIS2172

 *DOMAIN*: ARTIFICIAL INTELLIGENCE

 *DURATION*: 8 WEEKS

 *MENTOR*: NEELA SANTOSH


## Overview
This project implements Neural Style Transfer using a pretrained VGG19 convolutional neural network. The application blends the content of one image with the artistic style of another image to generate a new stylized output image.

Neural Style Transfer works by extracting content features from one image and style features from another image. The style is represented using Gram matrices, which capture texture and artistic patterns. The system then optimizes a new image by minimizing both content loss and style loss using gradient descent.

## Objective
- Load a pretrained CNN model (VGG19)
- Extract content and style features
- Apply Gram matrix for style representation
- Generate a stylized output image

## Features
- Neural Style Transfer using VGG19
- Content and style loss optimization
- Image optimization using gradient descent
- Saves final stylized image automatically

## Tech Stack
- Python 3.14
- PyTorch
- Torchvision
- PIL (Python Imaging Library)

## How to Run

Create virtual environment <br>
Install dependencies

### Run:
python main.py

Enter the content image file name when prompted.
Enter the style image file name when prompted.

The system processes both images and generates a stylized output image saved as:

output_styled.jpg

## Output 

