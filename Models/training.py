import torch
import torch.nn as nn
from architectures.mlp import MLP
from architectures.cnn import CNN, TinyCNN, SmallCNN, LargeCNN, CNNClassifier, LargeCNNClassifier, CNNClassifierDiopters
from architectures.cnnet import CNNET, CNNETConv
from torch.utils.tensorboard import SummaryWriter

import time

class Trainer:
    def __init__(self, model_type, batch_size=32, loss_fn=nn.MSELoss(), patience=10, activation=nn.ReLU()):
        """
        Initializes the Trainer object.

        Parameters:
        ----------

        model_type: str
            The type of model to use. Either 'mlp' or 'cnn'.
        batch_size: int
            The batch size to use for training.
        loss_fn: torch.nn loss function
            The loss function to use.
        patience: int
            The number of epochs to wait for early stopping.
        """

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type

        if model_type == 'mlp':
            print('Trainer initialized with MLP model.')
            self.model = MLP().to(self.device)
        elif model_type == 'cnn':
            print('Trainer initialized with CNN model.')
            self.model = CNN().to(self.device)
        elif model_type == 'tinycnn':
            print('Trainer initialized with Tiny CNN model.')
            self.model = TinyCNN().to(self.device)
        elif model_type == 'smallcnn':
            print('Trainer initialized with Small CNN model.')
            self.model = SmallCNN().to(self.device)
        elif model_type == 'largecnn':
            print('Trainer initialized with CNN model.')
            self.model = LargeCNN().to(self.device)
        elif model_type == 'cnnclassifier':
            print('Trainer initialized with CNN Classifier model.')
            self.model = CNNClassifier().to(self.device)
        elif model_type == 'largecnnclassifier':
            print('Trainer initialized with Large CNN Classifier model.')
            self.model = LargeCNNClassifier().to(self.device)
        elif model_type == 'cnnclassifierdiopters':
            print('Trainer initialized with CNN Classifier Diopters model.')
            self.model = CNNClassifierDiopters().to(self.device)
        elif model_type == 'cnnet':
            print('Trainer initialized with CNNET model.')
            self.model = CNNET(activation).to(self.device)
        elif model_type == 'cnnetconv':
            print('Trainer initialized with CNNETConv model.')
            self.model = CNNETConv(activation).to(self.device)
        else:
            raise ValueError('Model type not supported. Use MLP or CNN.')
        

        # adjust model parameters here
        #self.model = MLP().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.loss_fn = loss_fn
        self.patience = patience
        self.batch_size = batch_size

        print('Model initialized.')

    def train(self, train_loader, val_loader, n_epochs=100, filename='model.pt'):
        """
        Trains the model.

        Parameters:
        ----------
        train_loader: torch.DataLoader
            Dataloader for training data.
        val_loader: torch DataLoader
            Dataloader for validation data.
        n_epochs: int
            The number of epochs to train for.
        
        Returns:
        -------
        losses: list
            The training losses.
        vlosses: list
            The validation losses.
        """
        
        # get only the model name from the filename
        model_name = filename.split('.')[0]
        writer = SummaryWriter('runs/' + self.model_type + '/' + model_name)

        hparams = {
            'model_type': self.model_type,
            'batch_size': self.batch_size,
            'loss_fn': str(self.loss_fn),
            'patience': self.patience,
            'optimizer': str(self.optimizer),
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'nr_parameters': sum(p.numel() for p in self.model.parameters())
        }
        
        metrics = {}

        writer.add_hparams(hparams, metrics)


        best_val_loss = float('inf')
        best_model_state = None
        epochs_no_improve = 0
        losses = []
        vlosses = []
        num_batches = len(train_loader)

        print('Training model with loss function:', self.loss_fn, 'and optimizer:', self.optimizer)

        for epoch in range(n_epochs):
            self.model.train()
            times = []
            running_loss = 0.0
            for idx, (X_batch, y_batch) in enumerate(train_loader):
                print(f'Epoch {epoch+1}, Batch {idx+1}/{num_batches}', end='\r')
                # send to device
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                # measure time for one forward pass
                start = time.time()
                y_pred_batch = self.model(X_batch)
                end = time.time()
                times.append(end - start)

                loss = self.loss_fn(y_pred_batch, y_batch)
                self.optimizer.zero_grad(set_to_none=True) # setting to None zeros the memory, while zero_grad() does not
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
            # output average time for one forward pass
            print('Average time for forward pass:', sum(times) / len(times) / self.batch_size)

            avg_loss = running_loss / len(train_loader)
            losses.append(avg_loss)
            writer.add_scalar("Loss/train", avg_loss, epoch)
            #print(f'Epoch {epoch+1}, Training loss: {avg_loss}')
            
            self.model.eval()
            running_vloss = 0.0

            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)

                    outputs = self.model(data)
                    loss = self.loss_fn(outputs, target)
                    running_vloss += loss.item()

            
            avg_vloss = running_vloss / len(val_loader)
            writer.add_scalar("Loss/validation", avg_vloss, epoch)
            vlosses.append(avg_vloss)

            print(f'Epoch {epoch+1}, Training loss: {avg_loss}, Validation loss: {avg_vloss}')

            # Check if the validation loss improved
            if avg_vloss < best_val_loss:
                print(f'Validation loss improved from {best_val_loss:.4f} to {avg_vloss:.4f}')
                best_val_loss = avg_vloss
                best_model_state = self.model.state_dict()  # Save best model state
                epochs_no_improve = 0  # Reset early stopping counter
            else:
                print(f'Validation loss did not improve. Best was {best_val_loss:.4f}')
                epochs_no_improve += 1

            #print('Epoch:', epoch, 'Training loss:', running_loss, 'Validation loss:', loss.item())

            # Early stopping based on patience
            if epochs_no_improve >= self.patience:
                print("Early stopping triggered. Restoring best model.")
                self.model.load_state_dict(best_model_state)
                break

        self.model.load_state_dict(best_model_state)
        writer.flush()
        writer.close()

        return losses, vlosses, best_val_loss
    
    def predict(self, X):
        """
        Predicts the output for the given input.

        Parameters:
        ----------
        X: torch.Tensor
            The input data.
        
        Returns:
        -------
        y_pred: torch.Tensor
            The predicted output.
        """
        X = X.to(self.device)
        y_pred = self.model.predict(X)
        return y_pred

    
    def save_model(self, path):
        """
        Saves model to file.

        Parameters:
        ----------
        path: str
            The path to save the model to.
        """

        torch.save(self.model.state_dict(), path)
        print('Model saved to', path)

    def save_logs(self, path):
        """
        Saves logs to file.
        """

        with open(path, 'w') as f:
            f.write('Model: ' + str(self.model) + '\n')
            f.write('Optimizer: ' + str(self.optimizer) + '\n')
            f.write('Loss function: ' + str(self.loss_fn) + '\n')
            f.write('Patience: ' + str(self.patience) + '\n')


        print('Logs saved to', path)

    def load_model(self, path):

        self.model.load_state_dict(torch.load(path, weights_only=False))
        print('Model loaded from', path)
        
        return self.model
    
    