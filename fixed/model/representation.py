import torch
import copy
from tqdm import tqdm
import numpy as np
import sys
from sklearn.base import BaseEstimator
from torch.optim import Adam
from data_loader.graphdata import BatchDataSet
from metriclearner import MetricLearningModel

class RepresentationLearningModel(BaseEstimator):
    def __init__(self,
                 alpha=0.5, lambda1=0.5, lambda2=0.001, hidden_dim=256,  # Model Parameters
                 dropout=0.2, batch_size=64, balance=True,   # Model Parameters
                 num_epoch=100, max_patience=20,  # Training Parameters
                 print=False, num_layers=1
                 ):
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.dropout = dropout
        self.num_epoch = num_epoch
        self.max_patience = max_patience
        self.batch_size = batch_size
        self.balance = balance
        self.cuda = torch.cuda.is_available()
        self.print = print
        self.num_layers = num_layers
        if print:
            self.output_buffer = sys.stderr
        else:
            self.output_buffer = None
        pass

    def fit(self, train_x, train_y):
        self.train_wrapper(train_x, train_y)

    def train_wrapper(self, train_x, train_y):
        input_dim = train_x.shape[1]
        self.model = MetricLearningModel(
            input_dim=input_dim, hidden_dim=self.hidden_dim, alpha=self.alpha, lambda1=self.lambda1,
            lambda2=self.lambda2, dropout_p=self.dropout, num_layers=self.num_layers
        )
        self.optimizer = Adam(self.model.parameters())
        if self.cuda:
            self.model.cuda(device=0)
        self.dataset = BatchDataSet(self.batch_size, train_x.shape[1])
        for _x, _y in zip(train_x, train_y):
            if np.random.uniform() <= 0.1: # ?????
                self.dataset.add_data_entry(_x.tolist(), _y.item(), 'valid')
            else:
                self.dataset.add_data_entry(_x.tolist(), _y.item(), 'train')
        self.dataset.initialize_dataset(balance=self.balance, output_buffer=self.output_buffer)
        self.train(
            cuda_device=0 if self.cuda else -1,
            output_buffer=self.output_buffer
        )
        if self.output_buffer is not None:
            print('Training Complete', file=self.output_buffer)

    def train(self, cuda_device=-1, output_buffer=sys.stderr):
        if output_buffer is not None:
            print('Start Training', file=output_buffer)
        best_f1 = 0
        best_model = None
        patience_counter = 0
        train_losses = []
        try:
            for epoch_count in range(self.num_epochs):
                batch_losses = []
                num_batches = self.dataset.initialize_train_batches()
                output_batches_generator = range(num_batches)
                if output_buffer is not None:
                    output_batches_generator = tqdm(output_batches_generator)
                for _ in output_batches_generator:
                    self.model.train()
                    self.model.zero_grad()
                    self.optimizer.zero_grad()
                    features, targets, same_class_features, diff_class_features = self.dataset.get_next_train_batch()
                    if cuda_device != -1:
                        features = features.cuda(device=cuda_device)
                        targets = targets.cuda(device=cuda_device)
                        same_class_features = same_class_features.cuda(device=cuda_device)
                        diff_class_features = diff_class_features.cuda(device=cuda_device)
                    _, _, batch_loss = self.model(
                        example_batch=features, targets=targets,
                        positive_batch=same_class_features, negative_batch=diff_class_features
                    )
                    batch_losses.append(batch_loss.detach().cpu().item())
                    batch_loss.backward()
                    self.optimizer.step()
                epoch_loss = np.sum(batch_losses).item()
                train_losses.append(epoch_loss)
                if output_buffer is not None:
                    print('=' * 100, file=output_buffer)
                    print('After epoch %2d Train loss : %10.4f' % (epoch_count, epoch_loss), file=output_buffer)
                    print('=' * 100, file=output_buffer)
                if epoch_count % self.valid_every == 0:
                    valid_batch_count = self.dataset.initialize_valid_batches()
                    vacc, vpr, vrc, vf1 = self.model.evaluate_model(self.dataset.get_next_valid_batch, valid_batch_count, cuda_device, output_buffer)
                    if vf1 > best_f1:
                        best_f1 = vf1
                        patience_counter = 0
                        best_model = copy.deepcopy(self.model.state_dict())
                    else:
                        patience_counter += 1
                    if self.dataset.initialize_test_batches() != 0:
                        tacc, tpr, trc, tf1 = self.model.evaluate_model(self.dataset.get_next_test_batch, 
                                                                        self.dataset.initialize_test_batches(), 
                                                                        cuda_device,
                                                                        output_buffer=output_buffer
                        )
                        if output_buffer is not None:
                            print('Test Set:       Acc: %6.3f\tPr: %6.3f\tRc %6.3f\tF1: %6.3f' % \
                                (tacc, tpr, trc, tf1), file=output_buffer)
                            print('=' * 100, file=output_buffer)
                    if output_buffer is not None:
                        print('Validation Set: Acc: %6.3f\tPr: %6.3f\tRc %6.3f\tF1: %6.3f\tPatience: %2d' % \
                            (vacc, vpr, vrc, vf1, patience_counter), file=output_buffer)
                        print('-' * 100, file=output_buffer)
                    if patience_counter == self.max_patience:
                        if best_model is not None:
                            self.model.load_state_dict(best_model)
                            if cuda_device != -1:
                                self.model.cuda(device=cuda_device)
                        break
        except KeyboardInterrupt:
            if output_buffer is not None:
                print('Training Interrupted by User!')
            if best_model is not None:
                self.model.load_state_dict(best_model)
                if cuda_device != -1:
                    self.model.cuda(device=cuda_device)
        if self.dataset.initialize_test_batches() != 0:
            tacc, tpr, trc, tf1 = self.model.evaluate_model(self.dataset.get_next_test_batch,
                                                             self.dataset.initialize_test_batches(), 
                                                             cuda_device)
            if output_buffer is not None:
                print('*' * 100, file=output_buffer)
                print('*' * 100, file=output_buffer)
                print('Test Set: Acc: %6.3f\tPr: %6.3f\tRc %6.3f\tF1: %6.3f' % \
                    (tacc, tpr, trc, tf1), file=output_buffer)
                print('%f\t%f\t%f\t%f' % (tacc, tpr, trc, tf1))
                print('*' * 100, file=output_buffer)
                print('*' * 100, file=output_buffer)

    def predict(self, text_x):
        if not hasattr(self, 'dataset'):
            raise ValueError('Cannnot call predict or evaluate in untrained model. Train First!')
        self.dataset.clear_test_set()
        for _x in text_x:
            self.dataset.add_data_entry(_x.tolist(), 0, part='test')
        return self.model.predict_model(iterator_function=self.dataset.get_next_test_batch,
            _batch_count=self.dataset.initialize_test_batches(), cuda_device=0 if self.cuda else -1,
        )

    def predict_proba(self, text_x):
        if not hasattr(self, 'dataset'):
            raise ValueError('Cannnot call predict or evaluate in untrained model. Train First!')
        self.dataset.clear_test_set()
        for _x in text_x:
            self.dataset.add_data_entry(_x.tolist(), 0, part='test')
        return self.model.predict_proba(
             iterator_function=self.dataset.get_next_test_batch,
            _batch_count=self.dataset.initialize_test_batches(), cuda_device=0 if self.cuda else -1
        )

    def evaluate(self, text_x, test_y):
        if not hasattr(self, 'dataset'):
            raise ValueError('Cannnot call predict or evaluate in untrained model. Train First!')
        self.dataset.clear_test_set()
        for _x, _y in zip(text_x, test_y):
            self.dataset.add_data_entry(_x.tolist(), _y.item(), part='test')
        acc, pr, rc, f1 = self.model.evaluate_model(
            iterator_function=self.dataset.get_next_test_batch,
            _batch_count=self.dataset.initialize_test_batches(), cuda_device=0 if self.cuda else -1,
            output_buffer=self.output_buffer
        )
        return {
            'accuracy': acc,
            'precision': pr,
            'recall': rc,
            'f1': f1
        }

    def score(self, text_x, test_y):
        if not hasattr(self, 'dataset'):
            raise ValueError('Cannnot call predict or evaluate in untrained model. Train First!')
        self.dataset.clear_test_set()
        for _x, _y in zip(text_x, test_y):
            self.dataset.add_data_entry(_x.tolist(), _y.item(), part='test')
        _, _, _, f1 = self.model.evaluate_model( 
            iterator_function=self.dataset.get_next_test_batch,
            _batch_count=self.dataset.initialize_test_batches(), cuda_device=0 if self.cuda else -1,
            output_buffer=self.output_buffer
        )
        return f1
    