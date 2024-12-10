# README

## Project Overview
This project involves creating masked image datasets and training models using PyTorch and Swin Transformers for masked image modeling. It includes:
1. **Image Preprocessing**: Generating masked images with mosaic effects.
2. **Dataset Management**: Creating custom PyTorch datasets for handling original and masked images.
3. **Model Training and Evaluation**: Using Swin Transformers to reconstruct masked images and evaluate model performance.

## Files and Directories

### File Descriptions

1. **`image_pro.ipynb`**:
   - Processes images to create datasets with a fixed masking pattern.
   - Outputs the processed masked images into specific directories.
   - Key functions include:
     - `generate_random_block_positions`: Determines random positions for mosaic blocks.
     - `apply_random_mosaic_blocks`: Applies mosaic blocks with different colors to images.
     - `process_images_in_directory`: Processes entire directories of images, resizing them and applying mosaics.

2. **`image_pro_ren.ipynb`**:
   - Processes images to create datasets with random fixed masking patterns, ensuring the same total area as in `image_pro.ipynb`.
   - Allows adjustments for the total area of mosaic blocks, such as 50x50.

3. **`model1.ipynb`**:
   - Loads masked image datasets and trains models for masked image reconstruction.
   - Uses PyTorch to define data loaders, datasets, and models.
   - Switchable between multiple Transformer models (e.g., Swin Transformers).
   - Implements training and evaluation pipelines for masked image modeling.

### Directory Structure
```
project_root/
|
├── data/
│   └── dataset/
│       ├── Beagle/
│       ├── Boxer/
│       ├── Bulldog/
│       ├── Dachshund/
│       └── German_Shepherd/
|
├── new_dataset_5050_random/
│   └── (Generated directories for each breed and mosaic color)
|
├── image_pro.ipynb
├── image_pro_ren.ipynb
└── model1.ipynb
```

## Prerequisites

### Dependencies
- Python 3.8+
- PyTorch
- Transformers library (`transformers`)
- PIL (Pillow)
- OpenCV
- NumPy
- scikit-learn

Install dependencies using:
```bash
pip install torch torchvision transformers pillow opencv-python numpy scikit-learn
```

## Usage Instructions

### Step 1: Preprocess Images
Run `image_pro.ipynb` and/or `image_pro_ren.ipynb` to process the original images and generate the masked datasets.
1. Define the input and output directories.
2. Adjust parameters like:
   - `target_size`: Resized image dimensions.
   - `total_mosaic_size`: Total area covered by mosaic blocks.
   - `block_size`: Size of individual mosaic blocks.
3. Run the notebook to save processed images into the `new_dataset_5050_random` directory.

### Step 2: Load Datasets
In `model1.ipynb`:
1. Adjust directory paths for the original and masked datasets.
2. Initialize the `MaskedImageDataset` class and data loaders for training and testing.

### Step 3: Train and Evaluate the Model
1. Select a model (e.g., Swin Transformer) in `model1.ipynb`.
2. Run the training pipeline.
3. Evaluate the model's performance using metrics like Mean Squared Error (MSE).

## Customization
- **Dataset Creation**:
  Modify the `process_images_in_directory` function to change mosaic parameters or add new colors.
- **Model Selection**:
  Edit `model1.ipynb` to switch between different transformer models for reconstruction tasks.
- **Evaluation Metrics**:
  Add or modify evaluation criteria in `model1.ipynb` as needed.

## Example Commands
```bash
# Step 1: Preprocess images
python image_pro.ipynb

# Step 2: Train model
python model1.ipynb
```

## Contact
For questions or issues, please contact the project maintainer at [email@example.com].

