import sys, os
sys.path.insert(1, os.getcwd())

import torch
from torch import nn
import numpy as np
import copy
from tqdm import tqdm
from data_loader.graphdata import BatchDataSet
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class MetricLearningModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_p=0.2, alpha=0.5, lambda1=0.5, lambda2=0.001, num_layers=1):
        super(MetricLearningModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.internal_dim = int(hidden_dim / 2)
        self.dropout_p = dropout_p
        self.alpha = alpha
        self.layer1 = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=self.hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_p)
        )
        self.feature = nn.ModuleList([nn.Sequential(
            nn.Linear(in_features=self.hidden_dim, out_features=self.internal_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(in_features=self.internal_dim, out_features=self.hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_p),
        ) for _ in range(num_layers)])

        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.hidden_dim, out_features=2),
            nn.LogSoftmax(dim=-1)
        )
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.loss_function = nn.NLLLoss(reduction='none')
        # print(self.alpha, self.lambda1, self.lambda2, sep='\t', end='\t')

    def extract_feature(self, x):
        out = self.layer1(x)
        for layer in self.feature:
            out = layer(out)
        return out

    def forward(self, example_batch,
                targets=None,
                positive_batch=None,
                negative_batch=None):
        train_mode = (positive_batch is not None and
                      negative_batch is not None and
                      targets is not None)
        h_a = self.extract_feature(example_batch)
        y_a = self.classifier(h_a)
        probs = torch.exp(y_a)
        batch_loss = None
        if targets is not None:
            ce_loss = self.loss_function(input=y_a, target=targets)
            batch_loss = ce_loss.sum(dim=-1)
        if train_mode:
            h_p = self.extract_feature(positive_batch)
            h_n = self.extract_feature(negative_batch)
            dot_p = h_a.unsqueeze(dim=1) \
                .bmm(h_p.unsqueeze(dim=-1)).squeeze(-1).squeeze(-1)
            dot_n = h_a.unsqueeze(dim=1) \
                .bmm(h_n.unsqueeze(dim=-1)).squeeze(-1).squeeze(-1)
            mag_a = torch.norm(h_a, dim=-1)
            mag_p = torch.norm(h_p, dim=-1)
            mag_n = torch.norm(h_n, dim=-1)
            D_plus = 1 - (dot_p / (mag_a * mag_p))
            D_minus = 1 - (dot_n / (mag_a * mag_n))
            trip_loss = self.lambda1 * torch.abs((D_plus - D_minus + self.alpha))
            ce_loss = self.loss_function(input=y_a, target=targets)
            l2_loss = self.lambda2 * (mag_a + mag_p + mag_n)
            total_loss = ce_loss + trip_loss + l2_loss
            batch_loss = (total_loss).sum(dim=-1)
        return probs, h_a, batch_loss
    
    def model_train(self, 
        dataset: BatchDataSet, 
        optimizer: torch.optim.Optimizer, 
        num_epochs: int, 
        max_patience: int = 5,
        valid_every: int = 1, 
        cuda_device=-1, 
        output_buffer=sys.stderr
    ):
        """
        Trains a given MetricLearningModel on a provided DataSet using the specified optimizer.
        
        Args:
            model (MetricLearningModel): The model to be trained.
            dataset (DataSet): An object that provides training, validation, and test batches.
            optimizer (torch.optim.Optimizer): The optimizer for gradient updates.
            num_epochs (int): Maximum number of epochs to train.
            max_patience (int): Number of epochs to tolerate a lack of validation improvement before early stopping.
            valid_every (int): Frequency (in epochs) to run validation.
            cuda_device (int): GPU device ID; use -1 for CPU.
            output_buffer (IO): A file-like object to stream output (e.g. sys.stderr, sys.stdout, or None).
        
        Returns:
            None: The function modifies the model in-place. The best model state is loaded if early stopping is triggered.
        """
        
        # Print a starting message if an output buffer is provided
        if output_buffer is not None:
            print('Start Training', file=output_buffer)

        # Check types to ensure code correctness
        assert hasattr(dataset, 'initialize_train_batches') and hasattr(dataset, 'get_next_train_batch'), \
            "Dataset must expose methods to initialize and retrieve train batches."

        best_f1 = 0.0         # Keep track of the best F1 score found so far
        best_model_state = None
        patience_counter = 0  # For early stopping
        train_losses = []     # Keep track of the total training loss per epoch

        # Attempt to train, gracefully handle keyboard interrupts
        try:
            for epoch_count in range(num_epochs):
                # === Training Phase ===
                super(MetricLearningModel, self).train()  # Set model to training mode
                batch_losses = []

                # Initialize the dataset for training
                num_batches = dataset.initialize_train_batches()
                
                # Use tqdm for progress bar if output_buffer is valid
                if output_buffer is not None:
                    batches_iterator = tqdm(range(num_batches), desc=f"Epoch {epoch_count+1}/{num_epochs}")
                else:
                    batches_iterator = range(num_batches)

                for _ in batches_iterator:
                    # Zero out gradients for both the model and optimizer
                    self.zero_grad()
                    optimizer.zero_grad()
                    
                    # Fetch a batch of features and labels from the dataset
                    features, targets, same_class_features, diff_class_features = dataset.get_next_train_batch()

                    # If a GPU device is specified, move data to the device
                    if cuda_device != -1:
                        features = features.cuda(cuda_device)
                        targets = targets.cuda(cuda_device)
                        same_class_features = same_class_features.cuda(cuda_device)
                        diff_class_features = diff_class_features.cuda(cuda_device)

                    # Forward pass through the model
                    _, _, batch_loss = self(
                        example_batch=features, 
                        targets=targets,
                        positive_batch=same_class_features, 
                        negative_batch=diff_class_features
                    )

                    # Compute loss, accumulate, and backprop
                    batch_losses.append(batch_loss.detach().cpu().item())
                    batch_loss.backward()

                    # Update parameters
                    optimizer.step()

                # End of epoch => compute total loss for the epoch
                epoch_loss = np.sum(batch_losses).item()
                train_losses.append(epoch_loss)

                # Print training status
                if output_buffer is not None:
                    print('=' * 100, file=output_buffer)
                    print(f"Epoch {epoch_count+1} Training Loss: {epoch_loss:10.4f}", file=output_buffer)
                    print('=' * 100, file=output_buffer)

                # === Validation and Optional Test Phase ===
                if epoch_count % valid_every == 0:
                    # Initialize validation batches
                    valid_batch_count = dataset.initialize_valid_batches()

                    # Evaluate on the validation split
                    vacc, vpr, vrc, vf1 = self.evaluate_model(
                        dataset.get_next_valid_batch, valid_batch_count, cuda_device, output_buffer
                    )

                    # Track best F1 score for early stopping
                    if vf1 > best_f1:
                        best_f1 = vf1
                        patience_counter = 0
                        # Save the best model state so far
                        best_model_state = copy.deepcopy(self.state_dict())
                    else:
                        patience_counter += 1

                    # Optionally evaluate on any test set if available
                    test_batches = dataset.initialize_test_batches()
                    if test_batches != 0:
                        tacc, tpr, trc, tf1 = self.evaluate_model(
                            dataset.get_next_test_batch, test_batches, cuda_device, output_buffer
                        )
                        if output_buffer is not None:
                            print(f"Test Set:       Acc: {tacc:6.3f}\tPr: {tpr:6.3f}\tRc {trc:6.3f}\tF1: {tf1:6.3f}",
                                file=output_buffer)
                            print('=' * 100, file=output_buffer)

                    # Print validation performance
                    if output_buffer is not None:
                        print(f"Validation Set: Acc: {vacc:6.3f}\tPr: {vpr:6.3f}\tRc {vrc:6.3f}\tF1: {vf1:6.3f}"
                            f"\tPatience: {patience_counter:2d}", file=output_buffer)
                        print('-' * 100, file=output_buffer)

                    # Early stopping check
                    if patience_counter >= max_patience:
                        if best_model_state is not None:
                            self.load_state_dict(best_model_state)
                            if cuda_device != -1:
                                self.cuda(cuda_device)
                        break

        except KeyboardInterrupt:
            # Graceful exit on keyboard interrupt, restore best model if available
            if output_buffer is not None:
                print('Training Interrupted by User!', file=output_buffer)
            if best_model_state is not None:
                self.load_state_dict(best_model_state)
                if cuda_device != -1:
                    self.cuda(cuda_device)

        # Final evaluation on the test set (if any test batches exist)
        test_batches = dataset.initialize_test_batches()
        if test_batches != 0:
            tacc, tpr, trc, tf1 = self.evaluate_model(
                dataset.get_next_test_batch, test_batches, cuda_device
            )
            if output_buffer is not None:
                print('*' * 100, file=output_buffer)
                print('*' * 100, file=output_buffer)
                print(f"Test Set: Acc: {tacc:6.3f}\tPr: {tpr:6.3f}\tRc {trc:6.3f}\tF1: {tf1:6.3f}", file=output_buffer)
                print(f"{tacc}\t{tpr}\t{trc}\t{tf1}", file=output_buffer)
                print('*' * 100, file=output_buffer)
                print('*' * 100, file=output_buffer)

        # Optionally return training losses or other artifacts for further processing
        # return train_losses


    def predict_model(self, iterator_function, _batch_count, cuda_device):
        probs = self.predict_proba(iterator_function, _batch_count, cuda_device)
        return np.argmax(probs, axis=-1)


    def predict_proba(
        self, 
        iterator_function, 
        batch_count, 
        cuda_device=-1
    ):
        """
        Generates predicted probabilities for a given dataset.

        Args:
            iterator_function (Callable[[], Tuple[torch.Tensor, torch.Tensor]]):
                A function (or callable) that returns the next batch of features and targets.
            batch_count (int):
                Number of batches to process.
            cuda_device (int, optional):
                GPU device ID; use -1 for CPU. Defaults to -1.
        
        Returns:
            np.ndarray: A NumPy array containing the predicted probabilities for all batches.
        """
        # Set the model to evaluation mode
        self.eval()
        
        predictions = []
        
        with torch.no_grad():
            for _ in tqdm(range(batch_count), desc="Predicting"):
                # Get data batch
                features, targets = iterator_function()
                if cuda_device != -1:
                    features = features.cuda(cuda_device)
                
                # Run forward pass through the model
                probs, _, _ = self(example_batch=features)
                
                # Move probabilities to CPU (if needed) and convert to NumPy
                # 'probs' is a Tensor, so do '.cpu().numpy()' if on GPU
                probs_np = probs.detach().cpu().numpy()
                
                # Extend the 'predictions' list with these probabilities
                predictions.extend(probs_np)
        
        # (Optional) Return the model to training mode if you want to continue training immediately
        self.train()
        
        # Convert the Python list of predictions to a NumPy array and return
        return np.array(predictions)

    def evaluate_model(
        self, 
        iterator_function, 
        batch_count, 
        cuda_device, 
        output_buffer=sys.stderr
    ):
        """
        Evaluate the model on a given dataset iterator, returning performance metrics.
        
        Args:
            iterator_function (Callable[[], Tuple[torch.Tensor, torch.Tensor]]):
                A function returning features and targets for each batch.
            batch_count (int): Number of batches to evaluate.
            cuda_device (int): GPU device ID; use -1 for CPU.
            output_buffer (IO): A file-like object to which status is printed (optional).
        
        Returns:
            Tuple[float, float, float, float]:
                A 4-tuple containing (accuracy, precision, recall, F1) all in percentages.
        """

        # Optionally print the number of batches
        if output_buffer is not None:
            print(f"Evaluating on {batch_count} batches...", file=output_buffer)

        # Set the model to evaluation mode to disable dropout, etc.
        self.eval()

        predictions = []
        expectations = []

        # Optionally wrap the batch range in tqdm for a progress bar
        batch_generator = range(batch_count)
        if output_buffer is not None:
            batch_generator = tqdm(
                batch_generator, 
                desc="Evaluating", 
                file=output_buffer
            )

        # Disable gradient calculations for speed and memory use
        with torch.no_grad():
            for _ in batch_generator:
                features, targets = iterator_function()
                
                # Move data to GPU if needed
                if cuda_device != -1:
                    features = features.cuda(cuda_device)
                    targets = targets.cuda(cuda_device)
                
                # Forward pass (targets are optional in forward, only features needed here)
                probs, _, _ = self(example_batch=features)

                # Convert probabilities from GPU (if used) to CPU for NumPy operations
                probs_cpu = probs.detach().cpu().numpy()
                targets_cpu = targets.detach().cpu().numpy()

                # Predict class by taking argmax
                batch_pred = np.argmax(probs_cpu, axis=-1).tolist()
                batch_tgt = targets_cpu.tolist()

                # Collect predictions and true labels
                predictions.extend(batch_pred)
                expectations.extend(batch_tgt)

        # Return the model to training mode if you want to continue training right after
        self.train()

        # Compute metrics
        accuracy = accuracy_score(expectations, predictions) * 100
        precision = precision_score(expectations, predictions, average='binary') * 100
        recall = recall_score(expectations, predictions, average='binary') * 100
        f1 = f1_score(expectations, predictions, average='binary') * 100

        return accuracy, precision, recall, f1


    def show_representation(self, iterator_function, _batch_count, cuda_device, name, output_buffer=sys.stderr):
        self.eval()
        with torch.no_grad():
            representations = []
            expected_targets = []
            batch_generator = range(_batch_count)
            if output_buffer is not None:
                batch_generator = tqdm(batch_generator)
            for _ in batch_generator:
                iterator_values = iterator_function()
                features, targets = iterator_values[0], iterator_values[1]
                if cuda_device != -1:
                    features = features.cuda(device=cuda_device)
                _, repr, _ = self(example_batch=features)
                repr = repr.detach().cpu().numpy()
                print(repr.shape)
                representations.extend(repr.tolist())
                expected_targets.extend(targets.numpy().tolist())
            self.train()
            print(np.array(representations).shape)
            print(np.array(expected_targets).shape)
            #plot_embedding(representations, expected_targets, title=name)