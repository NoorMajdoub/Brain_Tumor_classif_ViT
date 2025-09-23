# Brain Tumor Classification with Vision Transformer (ViT)

A simple deep learning project for classifying brain tumor types from MRI images using a fine-tuned Vision Transformer model.

##  Problem Statement

Classify MRI brain scans into four categories:
- **Glioma Tumor**: Aggressive brain tumor requiring immediate attention
- **Meningioma Tumor**: Most common brain tumor type
- **No Tumor**: Healthy brain scan
- **Pituitary Tumor**: Tumor affecting the pituitary gland

##  Dataset

The project uses the Brain Tumor Classification MRI dataset containing:
- **Training Set**: ~2,870 images
- **Test Set**: 394 images
- **Image Size**: 224x224 pixels (RGB)
- **Classes**: 4 tumor types


## TO_DO
- Fix glimora class unbalances
## TO_experiment

- [ ] Implement focal loss for better imbalance handling
- [ ] Add ensemble methods with multiple models
- [ ] Use stratified k-fold cross-validation?
- [ ] Experiment with different ViT variants (ViT-Large, DeiT)
- [ ] Add explainability with attention visualization
- [ ] Implement confidence thresholds for clinical use
