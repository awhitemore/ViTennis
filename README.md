# 🎾 ViTennis

A computer vision application that uses Vision Transformer (ViT) to classify tennis player actions and find similar images using vector similarity search. This project demonstrates advanced representation learning and practical application of embedding-based search for interpretability and model analysis.

## 🎯 Project Overview

ViTennis combines the power of Vision Transformers with interactive similarity search to:
- **Classify tennis actions** into 4 categories: backhand, forehand, ready_position, serve
- **Find similar images** using semantic embeddings rather than just class labels
- **Visualize attention maps** to understand what the model focuses on
- **Enable label-free retrieval** to identify visual similarity beyond predictions

## ✨ Key Features

- **🎯 High Accuracy**: 91.25% validation accuracy on tennis action classification
- **🔍 Semantic Search**: Find visually similar images using ViT embeddings
- **👁️ Attention Visualization**: See what the model focuses on with attention overlays
- **🖥️ Interactive GUI**: User-friendly PyQt5 interface for easy exploration
- **⚡ Fast Retrieval**: Efficient KNN search over 2,000 image embeddings
- **🛡️ Robust Error Handling**: Graceful handling of problematic images

## 🏗️ System Architecture

```
User Input (Single image)
         ↓
ViT Encoder (fine-tuned)
         ↓
[CLS] Token Extraction → 384-D Vector
         ↓
KNN Search (scikit-learn)
         ↓
Top-5 Most Similar Frames
         ↓
GUI Display with Images + Metadata
```

## 🚀 Quick Start

### Option 1: Use Pre-trained Model (Recommended)

1. **Clone the repository**
   ```bash
   git clone https://github.com/awhitemore/ViTennis.git
   cd ViTennis
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the pre-trained model**
   - Go to [Releases](https://github.com/awhitemore/ViTennis/releases)
   - Download `tennis-model-v1.0.0.zip`
   - Extract to your project root (creates `temp/` folder)

4. **Run the application**
   ```bash
   python app_gui.py
   ```

### Option 2: Train Your Own Model

1. **Install dependencies**
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

4. **Run the application**
   ```bash
   python app_gui.py
   ```

## 🎮 Usage Guide

1. **Launch the application**: `python app_gui.py`
2. **Upload an image**: Click "Upload Image" and select a tennis image
3. **Select true class**: Choose the correct tennis action from the dropdown
4. **Explore results**: 
   - View the uploaded image with prediction and confidence
   - See 5 most similar images from the dataset
   - Toggle attention overlay to visualize model focus
   - Compare true vs predicted classes

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Training Loss** | 0.5201 |
| **Validation Accuracy** | 91.25% |
| **Validation Loss** | 0.2509 |
| **Training Time** | ~9 minutes on CPU |
| **Model Size** | ~22MB |

## 🧠 Technical Implementation

### Core Components

- **Vision Transformer (ViT)**
  - Fine-tuned on custom tennis dataset (2,000 images)
  - Extracts 384-dimensional `[CLS]` token embeddings
  - Used for both classification and semantic similarity

- **Scikit-learn NearestNeighbors**
  - Efficient KNN search over stored embeddings
  - L2 distance-based similarity matching
  - Replaces FAISS due to hardware compatibility

- **PyQt5 GUI Interface**
  - Interactive image upload and display
  - Real-time attention map visualization
  - Similarity search results presentation

### Key Functions

- `get_cls_token()`: Extract ViT embeddings from images
- `get_nearest_images()`: Perform KNN similarity search
- `show_vit_attention_on_image()`: Generate attention visualizations
- `generate_embeddings_and_metadata()`: Build vector index from dataset

## 📁 Project Structure

```
ViTennis/
├── app_gui.py          # Main GUI application
├── vit.py              # Model training script
├── util.py             # Utility functions (embeddings, attention)
├── dataset.py          # Dataset loading and preprocessing
├── requirements.txt    # Python dependencies
├── .gitignore         # Git ignore rules
├── data/              # Tennis dataset (not in repo)
│   └── images/
│       ├── backhand/
│       ├── forehand/
│       ├── ready_position/
│       └── serve/
└── temp/              # Trained model (generated/downloaded)
    └── 4-tennis/
```

## 🔧 Requirements

- **Python**: 3.8+
- **PyTorch**: 2.0.0+
- **Transformers**: 4.30.0+
- **PyQt5**: 5.15.0+
- **Other**: See `requirements.txt` for complete list

## 🎯 Professional Skills Demonstrated

### **Machine Learning Engineering**
- Vision Transformer fine-tuning and adaptation
- Embedding extraction and vector-based reasoning
- Model interpretability through attention visualization

### **System Design**
- Full ML pipeline from data preprocessing to GUI interface
- Robust error handling and graceful degradation
- Hardware-agnostic design with library flexibility

### **Problem Solving Under Constraints**
- Adapted from FAISS to scikit-learn for compatibility
- Preserved project value through abstraction
- Delivered working solution despite system constraints

## 🔄 Design Evolution

This project evolved through multiple iterations:

| Version | Focus | Outcome |
|---------|-------|---------|
| **V0** | ViT classifier with LLM tennis coach | Basic classification |
| **V1** | Attention map visualization | Model interpretability |
| **V2** | FAISS vector store | High-performance similarity search |
| **V3** | Scikit-learn KNN | Hardware-compatible similarity search |

Each pivot preserved core value while adapting to real-world constraints.

## 🚀 Future Enhancements

- **Failure Case Analysis**: Systematic identification of edge cases
- **Temporal ViT**: Multi-frame sequence support
- **Scaling**: Re-introduce FAISS for 10K+ embeddings
- **User Feedback**: Online embedding refinement
- **Visualization**: t-SNE/UMAP embedding clusters

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- Dataset: [Tennis Player Actions Dataset](https://www.kaggle.com/datasets/orvile/tennis-player-actions-dataset)
- Model: Facebook's DeiT-small Vision Transformer
- Libraries: Hugging Face Transformers, PyTorch, PyQt5

---

**Built with ❤️ for tennis and computer vision enthusiasts**
