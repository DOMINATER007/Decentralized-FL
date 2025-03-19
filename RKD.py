def response_based_kd(leader, student, epochs=10, batch_size=32, learning_rate=0.001, temperature=3.0, alpha=0.5):
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
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Load and prepare test data
    test_data = np.load(student.dataset_test)
    X_test, y_test = test_data['x'], test_data['y']
    X_test = torch.as_tensor(X_test, dtype=torch.float32).unsqueeze(1)
    y_test = torch.as_tensor(y_test, dtype=torch.long)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Set up loss functions and optimizer
    criterion_hard = nn.CrossEntropyLoss()
    criterion_soft = nn.KLDivLoss(reduction='batchmean')
    optimizer = optim.Adam(student.model.parameters(), lr=learning_rate)

    # Train student model
    student.model.train()
    leader.model.eval()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            
            # Get teacher predictions (softened logits)
            with torch.no_grad():
                teacher_outputs, _ = leader.model(inputs)
                teacher_probs = torch.nn.functional.softmax(teacher_outputs / temperature, dim=1)
            
            # Get student predictions
            student_outputs, _ = student.model(inputs)
            student_probs = torch.nn.functional.log_softmax(student_outputs / temperature, dim=1)
            
            # Compute loss
            loss_hard = criterion_hard(student_outputs, targets)
            loss_soft = criterion_soft(student_probs, teacher_probs) * (temperature ** 2)
            total_loss_batch = alpha * loss_hard + (1 - alpha) * loss_soft
            
            total_loss_batch.backward()
            optimizer.step()
            total_loss += total_loss_batch.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

    # Evaluate the student model
    y_pred, y_true = [], []
    student.model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs, _ = student.model(inputs)
            _, preds = torch.max(outputs, dim=1)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(targets.cpu().numpy())

    test_accuracy = accuracy_score(y_true, y_pred)
    student.accuracy_history_list.append(test_accuracy)
    print(f"\n***Distillation Accuracy : {test_accuracy}***********\n")
    return test_accuracy
