import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.model import TextTransformer
import time

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10, verbose=True, device='cpu'):
    """Train the model and return training history"""
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if verbose and (epoch + 1) % 2 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

    return history

def evaluate_model(model, data_loader, device='cpu'):
    """Evaluate model and return accuracy"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

def create_model_with_params(params, vocab_size, max_length, device='cpu'):
    """Create a model with given hyperparameters"""
    model = TextTransformer(
        vocab_size=vocab_size,
        embedding_dim=params['embedding_dim'],
        num_heads=params['num_heads'],
        num_layers=params['num_layers'],
        dim_feedforward=params['dim_feedforward'],
        num_classes=4,
        max_length=max_length,
        dropout=params['dropout']
    ).to(device)
    return model

def evaluate_hyperparameters(params, X_train, y_train, X_val, y_val,
                             vocab_size, max_length, epochs=5, verbose=False, device='cpu'):
    """
    Evaluate a set of hyperparameters by training a model
    Returns validation accuracy (fitness score)
    """
    # Create data loaders
    train_dataset = TensorDataset(
        torch.LongTensor(X_train),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.LongTensor(X_val),
        torch.LongTensor(y_val)
    )

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'])

    # Create and train model
    model = create_model_with_params(params, vocab_size, max_length, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

    # Train for fewer epochs in optimization
    history = train_model(model, train_loader, val_loader, criterion,
                         optimizer, epochs=epochs, verbose=verbose, device=device)

    # Return validation accuracy as fitness
    val_accuracy = history['val_acc'][-1]
    return val_accuracy, model
