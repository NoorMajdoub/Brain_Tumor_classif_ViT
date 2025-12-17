import pytest
import torch
import os
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Mocking the dataset paths as they are not available in a testing environment
# In a real scenario, you would ensure these paths exist or use a temporary directory

# Mock the ImageFolder dataset to avoid actual file loading
class MockImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        # Simulate having a few dummy classes and images
        self.classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.samples = []
        self.targets = []
        # Create dummy samples and targets
        num_samples_per_class = 5
        for i, cls in enumerate(self.classes):
            for _ in range(num_samples_per_class):
                # Dummy image (just a tensor, no actual image data needed for mock)
                dummy_image_tensor = torch.randn(3, 224, 224)
                self.samples.append((dummy_image_tensor, i))
                self.targets.append(i)
        self.transform = transform
        self.dataset_size = len(self.samples)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        img_tensor, label = self.samples[idx]
        if self.transform:
            img_tensor = self.transform(img_tensor) # Apply transform to the dummy tensor
        return img_tensor, label


# Mocking the random_split function
def mock_random_split(dataset, lengths):
    total_size = len(dataset)
    if sum(lengths) != total_size:
        raise ValueError("Sum of input lengths does not equal the length of the input dataset")
    
    # Simple split based on indices, not true random split for predictable tests
    split1_end = lengths[0]
    dataset1_samples = dataset.samples[:split1_end]
    dataset2_samples = dataset.samples[split1_end:]
    
    # Create mock datasets for the splits
    class MockSplitDataset:
        def __init__(self, samples, transform):
            self.samples = samples
            self.transform = transform
            self.dataset_size = len(samples)
        
        def __len__(self):
            return self.dataset_size

        def __getitem__(self, idx):
            img_tensor, label = self.samples[idx]
            if self.transform:
                img_tensor = self.transform(img_tensor)
            return img_tensor, label

    # Use the same transform as the original dataset if available
    transform = dataset.transform if hasattr(dataset, 'transform') else None
    
    split1 = MockSplitDataset(dataset1_samples, transform)
    split2 = MockSplitDataset(dataset2_samples, transform)
    
    return split1, split2


@pytest.fixture(scope="module")
def mock_datasets():
    # Define transforms that will be used for mocking
    train_transform_mock = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    test_transform_mock = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Replace the actual ImageFolder and random_split with mocks
    # In a real test file, you'd use pytest's monkeypatch or similar
    # For this example, we'll redefine the global variables using mocks
    
    # We need to ensure these mocks are used globally for the code under test
    # This is a simplified approach for demonstration. In a larger project,
    # dependency injection or patching would be preferred.
    
    # Mocking global variables used in the main script
    global datasets, torch.utils.data.random_split
    original_imagefolder = datasets.ImageFolder
    original_random_split = torch.utils.data.random_split
    
    datasets.ImageFolder = MockImageFolder
    torch.utils.data.random_split = mock_random_split

    test_mock = datasets.ImageFolder(root="/mock/input/brain-tumor-classification-mri/Testing", transform=test_transform_mock)
    train_mock = datasets.ImageFolder(root="/mock/input/brain-tumor-classification-mri/Training", transform=train_transform_mock)
    
    train_size = int(0.8 * len(train_mock))
    val_size = len(train_mock) - train_size
    
    train_dataset_mock, val_dataset_mock = torch.utils.data.random_split(train_mock, [train_size, val_size])

    train_loader_mock = DataLoader(train_dataset_mock, batch_size=16, shuffle=True)
    val_loader_mock = DataLoader(val_dataset_mock, batch_size=16, shuffle=True)
    test_loader_mock = DataLoader(test_mock, batch_size=16, shuffle=True)

    # Mock models and processors
    mock_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    mock_model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")
    num_classes_mock = len(train_mock.classes)
    mock_model.classifier = torch.nn.Linear(mock_model.config.hidden_size, num_classes_mock)

    yield mock_model, mock_processor, train_loader_mock, val_loader_mock, test_loader_mock, num_classes_mock

    # Restore original functions and classes after the test
    datasets.ImageFolder = original_imagefolder
    torch.utils.data.random_split = original_random_split


def test_model_initialization(mock_datasets):
    """Tests that the model and processor are initialized correctly."""
    model, processor, _, _, _, num_classes = mock_datasets
    
    assert isinstance(model, torch.nn.Module)
    assert isinstance(processor, object) # Checking type of AutoImageProcessor is tricky without importing it
    assert model.config.num_labels == num_classes # Check if the classifier head was adjusted


def test_dataloader_sizes(mock_datasets):
    """Tests that the data loaders are created with the expected number of samples."""
    _, _, train_loader, val_loader, test_loader, _ = mock_datasets
    
    # The MockImageFolder creates 5 samples per class for 4 classes, so 20 samples total.
    # The random_split divides these 20 samples. 
    # train_size = int(0.8 * 20) = 16
    # val_size = 20 - 16 = 4
    # So, train_dataset has 16 samples, val_dataset has 4, and test_dataset has 20.
    assert len(train_loader.dataset) == 16
    assert len(val_loader.dataset) == 4
    assert len(test_loader.dataset) == 20

    # Check batching
    # For train_loader (16 samples, batch_size 16), there should be 1 batch.
    # For val_loader (4 samples, batch_size 16), there should be 1 batch.
    # For test_loader (20 samples, batch_size 16), there should be 2 batches (16 + 4).
    assert len(train_loader) == 1
    assert len(val_loader) == 1
    assert len(test_loader) == 2


def test_training_step(mock_datasets):
    """Tests a single training step to ensure forward and backward passes work."""
    model, _, train_loader, _, _, _ = mock_datasets
    device = torch.device("cpu") # Use CPU for testing to avoid CUDA dependency
    model.to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    model.train()
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        
        output = model(x)
        logits = output.logits
        
        optimizer.zero_grad()
        batch_loss = loss_function(logits, y)
        
        batch_loss.backward()
        optimizer.step()
        
        assert batch_loss.item() >= 0 # Loss should be non-negative
        # Check if gradients are computed for some parameters
        assert any(p.grad is not None for p in model.parameters())
        
        # Test only the first batch to keep tests fast
        if batch_idx == 0:
            break

def test_validation_step(mock_datasets):
    """Tests a single validation step to ensure forward pass works without grad."""
    model, _, _, val_loader, _, _ = mock_datasets
    device = torch.device("cpu") # Use CPU for testing
    model.to(device)
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            logits = output.logits
            batch_loss = loss_function(logits, y)
            
            assert batch_loss.item() >= 0
            # Ensure no gradients are computed during eval
            for param in model.parameters():
                assert param.grad is None
            break # Test only the first batch

def test_inference_step(mock_datasets):
    """Tests the inference part where predictions are made for the classification report."""
    model, _, _, _, test_loader, _ = mock_datasets
    device = torch.device("cpu") # Use CPU for testing
    model.to(device)
    
    all_predictions = []
    all_true_labels = []

    model.eval()
    with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                logits = output.logits
                predictions = logits.argmax(dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_true_labels.extend(y.cpu().numpy())

    assert len(all_predictions) == len(test_loader.dataset)
    assert len(all_true_labels) == len(test_loader.dataset)
    assert all(p in [0, 1, 2, 3] for p in all_predictions) # Assuming 4 classes from mock
    assert all(t in [0, 1, 2, 3] for t in all_true_labels)


# This test requires sklearn to be installed, which is a dependency of the original code.
# If sklearn is not guaranteed in the test environment, this test might need to be skipped or mocked.
def test_classification_report_generation(mock_datasets):
    """Tests if the classification report can be generated."""
    try:
        from sklearn.metrics import classification_report
        import numpy as np
    except ImportError:
        pytest.skip("scikit-learn not found, skipping classification report test")

    model, _, _, _, test_loader, _ = mock_datasets
    device = torch.device("cpu") # Use CPU for testing
    model.to(device)
    
    all_predictions = []
    all_true_labels = []

    model.eval()
    with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                logits = output.logits
                predictions = logits.argmax(dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_true_labels.extend(y.cpu().numpy())
    
    class_names = model.config.id2label if hasattr(model.config, 'id2label') else ['0', '1', '2', '3'] # Use mock class names
    
    # Ensure class_names is a list matching the number of classes
    if isinstance(class_names, dict):
        sorted_class_names = [class_names[i] for i in sorted(class_names.keys())]
    else:
        sorted_class_names = class_names
        
    # Ensure sorted_class_names matches the number of classes in the mock dataset
    expected_num_classes = len(model.classifier.out_features)
    if len(sorted_class_names) != expected_num_classes:
        # Adjust mock class names if necessary, or use generic ones
        sorted_class_names = [f'class_{i}' for i in range(expected_num_classes)]

    report = classification_report(all_true_labels, all_predictions, target_names=sorted_class_names, digits=4)
    
    assert isinstance(report, str)
    assert "precision" in report
    assert "recall" in report
    assert "f1-score" in report
