import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

class CNNLSTM(nn.Module):
    """
    Applies 1D CNN to automatically extract features from EEG windows and passes them to an LSTM for classification.
    """
    def __init__(self, num_classes=3, in_channels=41, cnn_channels=16, lstm_hidden_size=32, lstm_layers=1, lr=1e-4, device=torch.device("cpu")):
        super(CNNLSTM, self).__init__()
        self.device = device  # Set device for the model

        # CNN Feature Extractor
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=cnn_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=cnn_channels * 2, out_channels=cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
        )

        # Fully Connected Classifier
        self.fc = nn.Sequential(
            nn.BatchNorm1d(lstm_hidden_size),
            nn.Linear(lstm_hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

        # apply L2 weight decay
        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()
        
        # TensorBoard writer
        self.writer = SummaryWriter()

        # move model to device, similarly for all tensors
        self.to(self.device)
    
    def forward(self, x):
        """
        x: (batch, 41, seq_length)
        """
        x = x.to(self.device)
        cnn_out = self.cnn(x)

        # Permute back to (batch, seq_length, features)
        cnn_out = cnn_out.permute(0, 2, 1)
        lstm_out, (h_n, _) = self.lstm(cnn_out)

        # Use the last hidden state from the final LSTM layer
        last_hidden = h_n[-1]

        output = self.fc(last_hidden)

        return output
    
    def train_step(self, x, y):
        """
        x: (batch, seq_length, 41)
        y: (batch)
        """
        x = x.to(self.device)
        y = y.to(self.device)
        self.optimizer.zero_grad()
        outputs = self(x)
        loss = self.criterion(outputs, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def save_checkpoint(self, epoch, filename):
        """
        Save a checkpoint where training can resume.
        """
        state = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(state, filename)
        print(f"Checkpoint saved to {filename}")
    
    def load_checkpoint(self, filename):
        """
        Load a checkpoint to resume training or perform inference.
        You should use this method if you want to perform inference on a trained model.
        """
        if os.path.isfile(filename):
            checkpoint = torch.load(filename, map_location=self.device, weights_only=True)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Checkpoint loaded from {filename} at epoch {checkpoint.get('epoch', 'Unknown')}")
        else:
            print(f"No checkpoint found at {filename}")
    
    def inference(self, dataloader):
        """
        Returns the predictions for the entire dataset
        actual labels 0, 1, or 2 not logits or probabilities
        does softmax and argmax automatically
        """
        self.to(self.device)
        self.eval()
        all_preds = []
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.to(self.device)
                outputs = self(x)
                preds = torch.softmax(outputs, dim=1).argmax(dim=1)
                all_preds.append(preds)
        return torch.cat(all_preds, dim=0)
    
    def inference_single(self, x):
        """
        Run inference on a single window
        x: (1, seq_length, 41) - single EEG window
        
        Returns the class prediction (0, 1, or 2)
        """
        self.to(self.device)
        self.eval()
        with torch.no_grad():
            x = x.to(self.device)
            outputs = self(x)
            pred = torch.softmax(outputs, dim=1).argmax(dim=1).item()
        return pred
    
    def train_model(self, train_loader, val_loader=None, epochs=100, checkpoint_interval=50):
        """
        Uses TensorBoard to log training loss by default.
        You can start this by running `tensorboard --logdir=runs` in the terminal.
        """
        for epoch in range(epochs):
            self.train()  # Set model to training mode
            total_loss = 0.0

            for i, (x, y) in enumerate(train_loader):
                loss = self.train_step(x, y)
                total_loss += loss
                global_step = epoch * len(train_loader) + i
                self.writer.add_scalar("Batch Loss", loss, global_step)
            avg_loss = total_loss / len(train_loader)
            self.writer.add_scalar("Epoch Loss", avg_loss, epoch+1)
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}")
            
            if val_loader is not None:
                self.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for x_val, y_val in val_loader:
                        x_val = x_val.to(self.device)
                        y_val = y_val.to(self.device)
                        outputs = self(x_val)
                        loss_val = self.criterion(outputs, y_val)
                        val_loss += loss_val.item()
                    avg_val_loss = val_loss / len(val_loader)
                    self.writer.add_scalar("Validation Loss", avg_val_loss, epoch+1)
                    print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {avg_val_loss:.4f}")
                    
            if (epoch + 1) % checkpoint_interval == 0:
                self.save_checkpoint(epoch+1, f"checkpoint_{epoch+1}.pth")

        self.writer.close()
        self.save_checkpoint(epochs, "final_model.pth")
        print("Training complete")

    def test_model(self, test_loader):
        """
        Performs inference on the test set and returns the predictions and actual labels.
        You should not use this method unless you want to calculate metrics manually.
        """
        self.to(self.device)
        self.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                outputs = self(x)
                preds = torch.softmax(outputs, dim=1).argmax(dim=1)
                all_preds.append(preds)
                all_labels.append(y)
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        return all_preds, all_labels
    
    def test_with_metrics(self, test_loader=None, all_preds=None, all_labels=None):
        """
        Pass in the test_loader or the predictions and labels directly.
        Returns common metrics for classification tasks:
        - Accuracy
        - Precision
        - Recall
        - F1 Score
        - Confusion Matrix
        """
        if test_loader is not None:
            all_preds, all_labels = self.test_model(test_loader)
        
        # Move tensors to CPU for numpy calculations
        preds = all_preds.cpu().numpy()
        labels = all_labels.cpu().numpy()
        
        # Ensure labels are within valid range
        num_classes = 3
        valid_mask = (labels < num_classes) & (labels >= 0)
        preds = preds[valid_mask]
        labels = labels[valid_mask]
        
        # Calculate metrics
        accuracy = (preds == labels).mean()
        
        # Initialize arrays for per-class metrics
        precision = np.zeros(num_classes)
        recall = np.zeros(num_classes)
        f1 = np.zeros(num_classes)
        
        # Calculate confusion matrix
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
        for pred, label in zip(preds, labels):
            confusion_matrix[label][pred] += 1
            
        # Calculate per-class metrics
        for i in range(num_classes):
            # Precision
            if confusion_matrix[:, i].sum() != 0:
                precision[i] = confusion_matrix[i,i] / confusion_matrix[:,i].sum()
            # Recall
            if confusion_matrix[i,:].sum() != 0:
                recall[i] = confusion_matrix[i,i] / confusion_matrix[i,:].sum()
            # F1
            if precision[i] + recall[i] != 0:
                f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
                
        # Calculate macro averages
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'confusion_matrix': confusion_matrix
        }
        
        return metrics
    