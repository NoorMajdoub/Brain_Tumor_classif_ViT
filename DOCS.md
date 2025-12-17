# Brain Tumor Classification with Vision Transformer

This script trains and evaluates a Vision Transformer (ViT) model for brain tumor classification using MRI images.

## Global Variables

- `device`: Specifies the computational device ('cuda' if available, otherwise 'cpu').
- `train_transform`: A sequence of transformations applied to training images (resizing, horizontal flip, normalization).
- `test_transform`: A sequence of transformations applied to testing images (resizing, normalization).
- `test`: An `ImageFolder` dataset for testing images.
- `train`: An `ImageFolder` dataset for training images.
- `train_dataset`, `val_dataset`: Datasets split from the original `train` dataset for training and validation.
- `train_loader`, `val_loader`, `test_loader`: `DataLoader` instances for batching and shuffling data.
- `processor`: An `AutoImageProcessor` for preprocessing images compatible with the ViT model.
- `model`: An `AutoModelForImageClassification` (ViT) fine-tuned for the task.
- `num_classes`: The total number of distinct tumor classes in the dataset.
- `loss_function`: `CrossEntropyLoss` for calculating the training loss.
- `optimizer`: `Adam` optimizer for updating model weights.
- `train_N`, `test_N`: Number of samples in the training and test datasets, respectively.
- `epochs`: The total number of training epochs.

## Functions/Classes

*(Note: The original script directly uses global variables and has the training loop inline. No standalone functions or classes were defined for training or evaluation logic. The documentation below describes the behavior of the inline code sections.)*

**Section: Dataset and DataLoader Initialization**

- **Role/Purpose**: Loads and preprocesses image data, then creates data loaders for training, validation, and testing.
- **Input Parameters**: None directly, relies on global configuration (paths, transforms, batch size).
- **Output / Return Value**: Populates global variables `train_loader`, `val_loader`, `test_loader`, `train_dataset`, `val_dataset`.

**Section: Model Initialization and Configuration**

- **Role/Purpose**: Loads a pre-trained Vision Transformer model and adjusts its classifier head to match the number of classes in the dataset.
- **Input Parameters**: None directly, relies on global configuration (model name, number of classes).
- **Output / Return Value**: Populates global variables `processor`, `model`, `num_classes`.

**Section: Training Loop**

- **Role/Purpose**: Iterates through the specified number of epochs, performing training and validation steps.
  - **Training Step**: For each batch in `train_loader`, it performs a forward pass, calculates the loss, performs a backward pass to compute gradients, and updates model weights using the optimizer.
  - **Validation Step**: After each training epoch, it evaluates the model on the `val_loader` without computing gradients to assess performance.
- **Input Parameters**: None directly, uses global variables for `model`, `train_loader`, `val_loader`, `loss_function`, `optimizer`, `device`, and `epochs`.
- **Output / Return Value**: Prints training and validation loss and accuracy for each epoch. Updates the weights of the global `model`.

**Section: Detailed Classification Report**

- **Role/Purpose**: Evaluates the trained model on the `test_loader` and generates a detailed classification report including precision, recall, and F1-score for each class.
- **Input Parameters**: None directly, uses global variables for `model`, `test_loader`, `device`.
- **Output / Return Value**: Prints a `classification_report` to the console.
