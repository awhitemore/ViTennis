# 🎾 ViTennis

A computer vision application that uses Vision Transformer (ViT) to classify tennis player actions and find similar images using vector similarity search.

## Features

- **Tennis Action Classification**: Classifies images into 4 categories: backhand, forehand, ready_position, serve
- **Attention Visualization**: Shows attention maps to understand what the model focuses on
- **Similar Image Search**: Finds the most similar images using vector similarity search
- **Interactive GUI**: PyQt5-based interface for easy image upload and exploration
- **High Accuracy**: 91.25% accuracy on validation set

## Quick Start

### Prerequisites

1. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download the dataset**
   - Download the tennis dataset from [Kaggle](https://www.kaggle.com/datasets/orvile/tennis-player-actions-dataset)
   - Extract to `data/images/` with structure:
     ```
     data/images/
     ├── backhand/
     ├── forehand/
     ├── ready_position/
     └── serve/
     ```

3. **Train the model**
   ```bash
   python vit.py
   ```
   This will create the trained model in `temp/4-tennis/` and generate embeddings.

4. **Run the application**
   ```bash
   python app_gui.py
   ```

## Usage

1. **Launch the application**: `python app_gui.py`
2. **Upload an image**: Click "Upload Image" and select a tennis image
3. **Select true class**: Choose the correct tennis action from the dropdown
4. **View results**: 
   - See the uploaded image with prediction
   - View 5 most similar images from the dataset
   - Toggle attention overlay to see what the model focuses on

## Technical Details

- **Model**: Vision Transformer (ViT) based on DeiT-small
- **Accuracy**: 91.25% on validation set
- **Dataset**: 2000 tennis action images across 4 classes
- **Vector Search**: Efficient similarity search using embeddings
- **GUI**: PyQt5 for cross-platform interface

## Requirements

- Python 3.8+
- See `requirements.txt` for full dependency list

## Model Performance

- **Training Loss**: 0.5201
- **Validation Accuracy**: 91.25%
- **Validation Loss**: 0.2509
- **Training Time**: ~9 minutes on CPU

## License

MIT License