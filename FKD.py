import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score


def get_penultimate_features(model, data_path):
    """
    Extract penultimate layer features from a trained model for a given dataset.
    
    Args:
        model: Trained PyTorch model (e.g., CNN3 or CNN7)
        data_path: Path to .npz file containing 'x' (inputs) and 'y' (labels)
    
    Returns:
        torch.Tensor: Penultimate layer features for the input data
    """
    data = np.load(data_path)
    X = data['x']
    X = torch.as_tensor(X, dtype=torch.float32).unsqueeze(1)  # Add channel dimension for MNIST (1, 28, 28)
    model.eval()
    with torch.no_grad():
        features = model.get_penultimate_weights(X)
    return features

def feature_based_kd(leader,student, epochs=10, batch_size=32, learning_rate=0.001, feature_loss_weight=1.0):
    import torch
    import numpy as np
    from torch import nn, optim
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.metrics import accuracy_score

    # Load and prepare train data
    train_data = np.load(student.dataset_train)
    X_train, y_train = train_data['x'], train_data['y']
    X_train = torch.as_tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_train = torch.as_tensor(y_train, dtype=torch.long)

    # Get teacher's penultimate features for train set
    teacher_features = torch.as_tensor(leader.penultimate_outputs, dtype=torch.float32)

# Ensure teacher_features has the correct shape
    if teacher_features.ndim == 1:
        teacher_features = teacher_features.unsqueeze(1)  # Add dimension if missing
    min_samples = min(X_train.shape[0], teacher_features.shape[0])
    teacher_features = teacher_features[:min_samples]
    X_train = X_train[:min_samples]
    y_train = y_train[:min_samples]    

# Ensure sizes match
    assert X_train.shape[0] == teacher_features.shape[0], "Mismatch in number of samples!"

# Now create the dataset safely
    train_dataset_with_features = TensorDataset(X_train, y_train, teacher_features)

    # Determine feature dimensions
    sample_input = X_train[0].unsqueeze(0)
    with torch.no_grad():
        student_outputs, student_features = student.model(sample_input)
        student_feature_dim = student_features.shape[1]
        teacher_feature_dim = teacher_features.shape[1] 

    # Create mapping layer if dimensions differ
    if student_feature_dim != teacher_feature_dim:
        mapping_layer = nn.Linear(teacher_feature_dim, student_feature_dim)
    else:
        mapping_layer = None

    # Create dataset with inputs, labels, and teacher features
    train_dataset_with_features = TensorDataset(X_train, y_train, teacher_features)
    train_loader_with_features = DataLoader(train_dataset_with_features, batch_size=batch_size, shuffle=True)

    # Set up losses and optimizer
    criterion_classification = nn.CrossEntropyLoss()
    criterion_feature = nn.MSELoss()
    if mapping_layer is not None:
        optimizer = optim.Adam(list(student.model.parameters()) + list(mapping_layer.parameters()), lr=learning_rate)
    else:
        optimizer = optim.Adam(student.model.parameters(), lr=learning_rate)

    # Train student model
    student.model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets, teacher_feat in train_loader_with_features:
            optimizer.zero_grad()
            student_outputs, student_features = student.model(inputs)
            loss_class = criterion_classification(student_outputs, targets)
            if mapping_layer is not None:
                mapped_teacher_features = mapping_layer(teacher_feat)
                loss_feature = criterion_feature(student_features, mapped_teacher_features)
            else:
                loss_feature = criterion_feature(student_features, teacher_feat)
            total_loss_batch = loss_class + feature_loss_weight * loss_feature
            total_loss_batch.backward()
            optimizer.step()
            total_loss += total_loss_batch.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader_with_features):.4f}")

    # Test the student model
    test_data = np.load(student.dataset_test)
    X_test, y_test = test_data['x'], test_data['y']
    X_test = torch.as_tensor(X_test, dtype=torch.float32).unsqueeze(1)
    y_test = torch.as_tensor(y_test, dtype=torch.long)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    y_pred, y_true = [], []
    student.model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs, _ = student.model(inputs)
            _, preds = torch.max(outputs, dim=1)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(targets.cpu().numpy())

    test_accuracy = accuracy_score(y_true, y_pred)
    #student.accuracy_history_list.append(test_accuracy)
    #print(f"\n***Distillation Accuracy : {test_accuracy}***********\n")
    return test_accuracy