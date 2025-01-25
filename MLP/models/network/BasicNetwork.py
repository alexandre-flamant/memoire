import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os


class BaseNetwork:
    def __init__(self, model, optimizer, loss_fn, device="cpu", logging=True, model_name=None):
        """
        Initialize the Base Network class.
        Args:
            model: PyTorch model (e.g., MLP or PINN).
            optimizer: Optimizer for model training (e.g., Adam).
            loss_fn: Loss function for the task.
            device: Device to run the model on ('cpu' or 'cuda').
            logging: Boolean to enable/disable TensorBoard logging (default: True).
            model_name: Custom name for the model logs. Defaults to 'class_name_YYMMDD_HHMM'.
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

        # Logging setup
        self.logging = logging
        if self.logging:
            timestamp = datetime.now().strftime("%y%m%d_%H%M")
            self.model_name = model_name if model_name else f"{model.__class__.__name__}_{timestamp}"
            log_dir = os.path.join(self.model_name, "log")
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = None

        # Internal counters
        self.epoch_count = 0

    def train_one_epoch(self, train_loader, log_graphic_name="loss"):
        """
        Train the model for one epoch and log batch losses.
        Args:
            train_loader: DataLoader for training data.
            log_graphic_name: Name prefix for TensorBoard logs.
        Returns:
            Average loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Forward pass
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_loss = loss.item()
            total_loss += batch_loss

            # Log batch loss to TensorBoard
            if self.logging:
                self.writer.add_scalar(f"{self.model_name}/log/{log_graphic_name}_train", batch_loss,
                                       self.epoch_count * len(train_loader) + batch_idx)

        average_loss = total_loss / len(train_loader)
        return average_loss

    def validate(self, val_loader, log_graphic_name="loss"):
        """
        Validate the model and log batch losses.
        Args:
            val_loader: DataLoader for validation data.
            log_graphic_name: Name prefix for TensorBoard logs.
        Returns:
            Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)

                batch_loss = loss.item()
                total_loss += batch_loss

                # Log batch loss to TensorBoard
                if self.logging:
                    self.writer.add_scalar(f"{self.model_name}/log/{log_graphic_name}_val", batch_loss,
                                           self.epoch_count * len(val_loader) + batch_idx)

        average_loss = total_loss / len(val_loader)
        return average_loss

    def fit(self, train_loader, val_loader=None, epochs=10, log_graphic_name="loss"):
        """
        Full training routine over multiple epochs.
        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data (optional).
            epochs: Number of training epochs.
            log_graphic_name: Name prefix for TensorBoard logs.
        """
        for epoch in range(epochs):
            self.epoch_count += 1
            train_loss = self.train_one_epoch(train_loader, log_graphic_name)
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}")

            # Log epoch-level loss
            if self.logging:
                self.writer.add_scalar(f"{self.model_name}/log/{log_graphic_name}_train_epoch", train_loss,
                                       self.epoch_count)

            if val_loader:
                val_loss = self.validate(val_loader, log_graphic_name)
                print(f"Validation Loss: {val_loss:.4f}")

                if self.logging:
                    self.writer.add_scalar(f"{self.model_name}/log/{log_graphic_name}_val_epoch", val_loss,
                                           self.epoch_count)

    def save_model(self, path):
        """
        Save the model weights.
        Args:
            path: Path to save the model.
        """
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """
        Load the model weights.
        Args:
            path: Path to load the model from.
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from {path}")

    def close_writer(self):
        """
        Close the TensorBoard writer.
        """
        if self.logging and self.writer:
            self.writer.close()
