import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class BayesianRegressionTrainer:
    """
    Trainer class for Bayesian Neural Networks for regression tasks.
    """
    
    def __init__(self, model, criterion=nn.MSELoss(), learning_rate=0.01, device=None):
        """
        Initialize the trainer.
        
        Args:
            model: The Bayesian regressor model
            criterion: Loss function (default: MSELoss)
            learning_rate: Learning rate for optimizer
            device: Device to run on (cpu/cuda)
        """
        self.model = model
        self.criterion = criterion
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.train_losses = []
        self.val_losses = []
        
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              batch_size=16, epochs=500, samples_nbr=3, complexity_cost_weight=None):
        """
        Train the Bayesian Neural Network.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            batch_size: Batch size for training
            epochs: Number of epochs to train
            samples_nbr: Number of forward passes per batch for ELBO loss
            complexity_cost_weight: Weight for the complexity cost (if None, uses 1/n)
            
        Returns:
            tuple: (train_losses, val_losses)
        """
        # Convert to tensors if they aren't already
        if not isinstance(X_train, torch.Tensor):
            X_train = torch.tensor(X_train).float()
            y_train = torch.tensor(y_train).float()
            
        if X_val is not None and not isinstance(X_val, torch.Tensor):
            X_val = torch.tensor(X_val).float()
            y_val = torch.tensor(y_val).float()
            
        # Default complexity cost weight
        if complexity_cost_weight is None:
            complexity_cost_weight = 1 / X_train.shape[0]
            
        # Create dataloaders
        train_ds = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        
        if X_val is not None:
            val_ds = torch.utils.data.TensorDataset(X_val, y_val)
            val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)
            
        # Training loop
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Compute ELBO loss
                loss = self.model.sample_elbo(
                    inputs=inputs,
                    labels=targets,
                    criterion=self.criterion,
                    sample_nbr=samples_nbr,
                    complexity_cost_weight=complexity_cost_weight
                )
                
                # Backpropagation
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                
            avg_train_loss = train_loss / len(train_loader)
            self.train_losses.append(avg_train_loss)
            
            # Validation
            if X_val is not None:
                val_loss = self._evaluate(val_loader)
                self.val_losses.append(val_loss)
                
                if (epoch + 1) % 100 == 0 or epoch == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                if (epoch + 1) % 100 == 0 or epoch == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}")
        
        return self.train_losses, self.val_losses
    
    def _evaluate(self, dataloader):
        """
        Evaluate the model on a dataloader.
        
        Args:
            dataloader: PyTorch DataLoader
            
        Returns:
            float: Average loss
        """
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Single forward pass for evaluation
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()
                
        return val_loss / len(dataloader)
    
    def predict(self, X, num_samples=100):
        """
        Make predictions with uncertainty estimates.
        
        Args:
            X: Input features
            num_samples: Number of forward passes for uncertainty estimation
            
        Returns:
            tuple: (means, stds, predictions)
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X).float()
            
        X = X.to(self.device)
        
        # Get predictions with uncertainty
        mean_pred, std_pred, all_preds = self.model.predict_with_uncertainty(X, num_samples=num_samples)
        
        return mean_pred.cpu().numpy(), std_pred.cpu().numpy(), all_preds.cpu().numpy()
    
    def plot_loss(self):
        """
        Plot the training and validation loss curves.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        
        if self.val_losses:
            plt.plot(self.val_losses, label='Validation Loss')
            
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.show() 