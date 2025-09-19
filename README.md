# Expert-Guided Explainable Few-Shot Learning for Medical Image Diagnosis

This repository contains the implementation of **GAIN (Guided Attention Inference Network)**, an expert-guided explainable few-shot learning framework for medical image diagnosis. The method integrates radiologist-provided regions of interest (ROIs) into model training to simultaneously enhance classification performance and interpretability.

## 🔍 Overview

Medical image analysis often faces challenges due to limited expert-annotated data. Our framework addresses this by:

- **Expert-Guided Learning**: Incorporates radiologist-provided ROIs as spatial supervision
- **Few-Shot Architecture**: Uses prototypical networks for learning with limited data
- **Explainable AI**: Employs Grad-CAM for generating interpretable attention maps
- **Joint Optimization**: Combines classification and explanation losses for improved performance

### Dependencies
- Python 3.7-3.10
- PyTorch 1.12.0+
- torchvision 0.13.0+
- NumPy 1.21.0+
- Pandas 1.3.0+
- Matplotlib 3.5.0+
- Pillow 8.3.0+
- pydicom 2.2.0+
- scikit-learn 1.0.0+

### Supported Datasets
#### 1. BraTS (Brain Tumor MRI)
#### 2. VinDr-CXR (Chest X-ray)

### Configuration
Update the dataset paths in the `Config` class:

```python
class Config:
    # Update these paths for your local setup
    IMG_DIR = './datasets/vindr_cxr/train'
    CSV_LABELS = './datasets/vindr_cxr/train.csv'
    
    # Model parameters
    IMG_SIZE = 224
    BATCH_SIZE = 8
    ALPHA = 0.10  # Explanation loss weight
    NUM_EPOCHS = 7
    # ... other parameters
```

## 🚀 Usage

### Training
Run the complete training pipeline:

```bash
python filename.py
```

### Key Training Parameters
- **N_SHOT**: Number of support examples per class 
- **N_QUERY**: Number of query examples per class
- **ALPHA**: Weight for explanation loss 
- **NUM_EPOCHS**: Training epochs 
- **BATCH_SIZE**: Batch size 



### Inference
```python
# Load trained model
checkpoint = torch.load('output/models/best_few_shot_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
prototypes = checkpoint['prototypes']

# Generate predictions and explanations
# See example usage in the main script
```

## 📈 Results

### Performance Improvements
| Dataset | Non-Guided | Guided | Improvement |
|---------|------------|--------|-------------|
| BraTS | 77.09% | 83.61% | +6.52% |
| VinDr-CXR | 54.33% | 73.29% | +18.96% |

### Key Findings
- **Optimal α value**: 0.10 provides the best balance between classification and explanation alignment
- **Attention Alignment**: Guided models consistently focus on diagnostically relevant regions
- **Clinical Relevance**: Improved interpretability enhances clinical trustworthiness




### Hyperparameter Tuning
Key parameters to adjust:
- `ALPHA`: Controls explanation supervision strength
- `BATCH_SIZE`: Adjust based on GPU memory
- `N_SHOT`, `N_QUERY`: Few-shot learning configuration
- `NUM_EPOCHS`, `NUM_EPISODES`: Training duration

## 📝 Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{uddin2025expert,
  title={Expert-Guided Explainable Few-Shot Learning for Medical Image Diagnosis},
  author={Uddin, Ifrat Ikhtear and Wang, Longwei and Santosh, KC},
  journal={arXiv preprint arXiv:2509.08007},
  year={2025}
}
```

## 🙏 Acknowledgments

This work was supported in part by:
- National Science Foundation under Award #2346643
- National Research Platform (NRP) Nautilus HPC cluster