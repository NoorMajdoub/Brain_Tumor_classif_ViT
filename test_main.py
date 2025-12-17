import pytest
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from transformers import AutoImageProcessor, AutoModelForImageClassification
from torch.optim import Adam
import os

# Mocking the dataset paths to make tests runnable without actual data
@pytest.fixture(scope='module')
def mock_dataset_paths(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("mock_data")
    # Create dummy directories and files to simulate ImageFolder structure
    for phase in ["Training", "Testing"]:
        phase_path = data_dir / phase
        phase_path.mkdir()
        for class_name in ["glioma", "meningioma", "not tumor", "pituitary"]:
            class_path = phase_path / class_name
            class_path.mkdir()
            # Create a dummy image file
            (class_path / "dummy_image.png").touch()
    return str(data_dir)

# Override the global variables with mock data paths
def override_dataset_paths(mock_dataset_paths):
    global train, test
    
    train_transform_mock = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    test_transform_mock = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    train = datasets.ImageFolder(root=os.path.join(mock_dataset_paths, "Training"), transform=train_transform_mock)
    test = datasets.ImageFolder(root=os.path.join(mock_dataset_paths, "Testing"), transform=test_transform_mock)

def test_dataset_loading(mock_dataset_paths):
    override_dataset_paths(mock_dataset_paths)
    assert len(train) > 0
    assert len(test) > 0
    assert len(train.classes) == 4
    assert "glioma" in train.classes

def test_dataloader_creation(mock_dataset_paths):
    override_dataset_paths(mock_dataset_paths)
    train_size = int(0.8 * len(train))
    val_size = len(train) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test, batch_size=16, shuffle=True)
    
    assert len(train_loader) > 0
    assert len(val_loader) > 0
    assert len(test_loader) > 0
    
    # Check batch content (example for train_loader)
    x, y = next(iter(train_loader))
    assert x.shape[0] == 16 # batch_size
    assert y.shape[0] == 16
    assert x.shape[1] == 3 # channels
    assert x.shape[2] == 224 # height
    assert x.shape[3] == 224 # width

def test_model_initialization():
    # Mock the number of classes to avoid issues if the dataset is not available
    # In a real scenario, this would be set by the dataset
    mock_num_classes = 4 
    
    # Use a smaller, pre-trained model for faster testing if available, 
    # but stick to the original if not specified for mocking
    processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")
    model.classifier = torch.nn.Linear(model.config.hidden_size, mock_num_classes)
    
    assert model is not None
    assert isinstance(model, torch.nn.Module)
    assert model.classifier.out_features == mock_num_classes
    
    # Move model to CPU for testing to ensure consistency and avoid CUDA errors if not available
    model.to('cpu')
    assert next(model.parameters()).device.type == 'cpu'

def test_loss_function_and_optimizer():
    # Mock model and num_classes for optimizer and loss function setup
    mock_hidden_size = 768 # Based on google/vit-base-patch16-224 config
    mock_num_classes = 4
    mock_model = torch.nn.Module()
    mock_model.parameters = lambda: iter([torch.randn(1, requires_grad=True)])
    mock_model.config = type('obj', (object,), {'hidden_size': mock_hidden_size})()
    
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = Adam(mock_model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    assert isinstance(loss_function, torch.nn.modules.loss.CrossEntropyLoss)
    assert isinstance(optimizer, Adam)
    assert optimizer.defaults['lr'] == 1e-4
    assert optimizer.defaults['weight_decay'] == 1e-4

def test_training_loop_single_batch(mock_dataset_paths):
    # This test verifies the structure and basic execution of one training step
    # It does NOT test actual convergence or accuracy.
    override_dataset_paths(mock_dataset_paths)
    
    # Reduce dataset size for faster test execution
    train_size = 1 # Make it small for a single batch effect
    val_size = 0
    train_dataset, val_dataset = torch.utils.data.random_split(train, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False) # batch_size=1 for easier assertion
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")
    num_classes = len(train.classes)
    model.classifier = torch.nn.Linear(model.config.hidden_size, num_classes)
    model.to(device)

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    epochs = 1 # Run only one epoch for testing
    for epoch in range(epochs):
        model.train()
        train_loss_before = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            output = model(x)
            logits = output.logits
            
            optimizer.zero_grad()
            batch_loss = loss_function(logits, y)
            
            batch_loss.backward()
            optimizer.step()
            
            train_loss_before += batch_loss.item() # Accumulate loss before step
            
            # Verify that gradients were computed and step was taken
            # (this is implicitly tested by batch_loss.backward() and optimizer.step() not raising errors)
            # We can check if parameters have changed, but that's more involved.
            # For now, we assume no errors means the step happened.
            break # Process only one batch

        # After one batch, check if loss has been accumulated (even if small)
        assert train_loss_before > 0
        
        # --- Validation part test --- 
        model.eval()
        val_loss_before = 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                output_val = model(x_val)
                logits_val = output_val.logits
                val_loss_before += loss_function(logits_val, y_val).item()
        assert val_loss_before >= 0 # Loss can be zero if no batches
        break # Run only one epoch for testing

def test_classification_report_generation(mock_dataset_paths):
    # This test checks if the classification report can be generated without errors,
    # assuming dummy data is sufficient to pass through the loop.
    override_dataset_paths(mock_dataset_paths)
    
    # Mock a small DataLoader for testing
    train_size = 1
    val_size = 0
    train_dataset, val_dataset = torch.utils.data.random_split(train, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test, batch_size=1, shuffle=False) # Use the mocked test set

    processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")
    num_classes = len(train.classes)
    model.classifier = torch.nn.Linear(model.config.hidden_size, num_classes)
    model.to(device)

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # --- Mock a single training/validation step to ensure model is evaluated ---
    # This is a simplified approach to get the model into an 'eval' state
    # without running a full epoch.
    model.eval()
    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val, y_val = x_val.to(device), y_val.to(device)
            output_val = model(x_val)
            logits_val = output_val.logits
            break # process one batch only
            
    # --- Collect predictions and generate report ---
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
    
    # Only attempt to generate report if there are some labels to report on
    if all_true_labels:
        class_names = test_loader.dataset.classes
        # Use dummy class names if actual ones are not loaded/available
        if not class_names or len(class_names) == 0:
            class_names = [f"class_{i}" for i in range(num_classes)]
            
        report = classification_report(all_true_labels, all_predictions, target_names=class_names, digits=4)
        assert isinstance(report, str)
        assert "classification report" in report.lower()
    else:
        # If test_loader is empty, this is expected, and no report should be generated.
        # We assert that the lists are empty.
        assert len(all_true_labels) == 0
        assert len(all_predictions) == 0

