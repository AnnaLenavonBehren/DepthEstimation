import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset, Subset
from training import Trainer
from data.data_mlp import DataMLP
from data.data_cnn import DataCNN, ETData
from data.data_cnn_diopters import DataCNNDiopters
from data.data_cnn_etdata import DataCNNET
from evaluation import EvaluateModel as eval
import torch.nn as nn
import matplotlib.pyplot as plt
from enum import Enum
import os
import numpy as np

# Hyperparameter

batch_size = 64
num_workers = 12
num_epochs = 100

model_nr = 500



# do not adjust these parameters

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#lrs = [10**-1, 10**-2, 10**-3, 10**-4, 10**-5, 10**-6]
#model_types = ['largecnnclassifier', 'cnnclassifier', 'cnn', 'largecnn', 'tinycnn', 'smallcnn']

# add all learning rates and model types you want to test, choose activation function
lrs = [10**-3]
model_types = ['cnnclassifier']
activation = nn.ReLU() # nn.Tanh()


# set to False if you only want to evaluate a model
train = True

# set to True if you only want to test the performance on one subject
test_only_one_subj = False


# train or only evaluate model


# if train == False, assign model_type
#filename = 'cnn_32.pt'
feature_permutation = ETData.NONE
nr_permutations = 20



if feature_permutation != ETData.NONE:
    train = False


perm_acc = 0
acc_id = 0

def __main__():
    global filename, feature_permutation, perm_acc
    print('Using device:', device)
    num_cores = os.cpu_count()
    print('Number of CPU cores:', num_cores)
    # Initialize data and model

    # hyperparameter search
    v_losses = []
    for lr in lrs:
        for model_type in model_types:
            filename = model_type + '_' + str(model_nr) + '.pt'

            for id in range(30, 44):
                acc_id = id
                for nr in range(nr_permutations):
                    perm_acc = nr

                    if model_type == 'mlp':
                        print('Using MLP model.')

                        data = DataMLP(test_only_one_subj=test_only_one_subj, subj_id=id)
                        model = Trainer(model_type)
                        total_params = sum(p.numel() for p in model.model.parameters())
                        print(f"Number of parameters: {total_params}")
                        
                        # set model parameters
                        model.optimizer = torch.optim.Adam(model.model.parameters(), lr=lr)
                        #model.optimizer = torch.optim.SGD(model.model.parameters(), lr=lr, momentum=0.9)
                        model.loss_fn = nn.MSELoss()

                        # define data loader
                        train_loader = DataLoader(TensorDataset(data.X_train, data.y_train), num_workers=num_workers, batch_size=batch_size, shuffle=True, persistent_workers=True)
                        val_loader = DataLoader(TensorDataset(data.X_val, data.y_val), num_workers=num_workers, batch_size=batch_size, shuffle=False, persistent_workers=True)
                        test_loader = DataLoader(TensorDataset(data.X_test, data.y_test), num_workers=num_workers, batch_size=batch_size, shuffle=False, persistent_workers=True)

                    elif model_type == 'cnn' or model_type == 'tinycnn' or model_type == 'smallcnn' or model_type == 'largecnn' or model_type == 'cnnclassifier' or model_type == 'largecnnclassifier' or model_type == 'cnnclassifierdiopters':
                        print('Using CNN model.')

                        data = DataCNN(test_only_one_subj=test_only_one_subj, subj_id=id, permutation=feature_permutation, seed=nr)
                        ds = data.ds
                        train_indices = data.train_indices
                        val_indices = data.val_indices
                        test_indices = data.test_indices

                        model = Trainer(model_type)
                        # get number of parameters
                        total_params = sum(p.numel() for p in model.model.parameters())
                        print(f"Number of parameters: {total_params}")
                        # set model parameters
                        model.optimizer = torch.optim.Adam(model.model.parameters(), lr=lr)
                        #model.optimizer = torch.optim.SGD(model.model.parameters(), lr=lr, momentum=0.9)
                        model.loss_fn = nn.MSELoss()

                        # create dataloaders
                        train_loader = DataLoader(Subset(ds, train_indices), batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
                        val_loader = DataLoader(Subset(ds, val_indices), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
                        test_loader = DataLoader(Subset(ds, test_indices), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
                    
                    elif model_type == 'cnnclassifierdiopters':
                        print('Using CNN model training on diopters.')

                        data = DataCNNDiopters()
                        ds = data.ds
                        train_indices = data.train_indices
                        val_indices = data.val_indices
                        test_indices = data.test_indices

                        model = Trainer(model_type)
                        # get number of parameters
                        total_params = sum(p.numel() for p in model.model.parameters())
                        print(f"Number of parameters: {total_params}")
                        # set model parameters
                        model.optimizer = torch.optim.Adam(model.model.parameters(), lr=lr)
                        #model.optimizer = torch.optim.SGD(model.model.parameters(), lr=lr, momentum=0.9)
                        model.loss_fn = nn.MSELoss()

                        # create dataloaders
                        train_loader = DataLoader(Subset(ds, train_indices), batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
                        val_loader = DataLoader(Subset(ds, val_indices), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
                        test_loader = DataLoader(Subset(ds, test_indices), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

                    elif model_type == 'cnnet' or model_type == 'cnnetconv':
                        print('Using CNN ET model.')

                        data = DataCNNET(test_only_one_subj=test_only_one_subj, subj_id=id)
                        ds = data.ds
                        train_indices = data.train_indices
                        val_indices = data.val_indices
                        test_indices = data.test_indices

                        model = Trainer(model_type, activation=activation)
                        # get number of parameters
                        total_params = sum(p.numel() for p in model.model.parameters())
                        print(f"Number of parameters: {total_params}")
                        # set model parameters

                        model.optimizer = torch.optim.Adam(model.model.parameters(), lr=lr)
                        model.loss_fn = nn.MSELoss()

                        # create dataloaders
                        train_loader = DataLoader(Subset(ds, train_indices), batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
                        val_loader = DataLoader(Subset(ds, val_indices), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
                        test_loader = DataLoader(Subset(ds, test_indices), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
                    
                    else:
                        print('Model type not supported. Use MLP or CNN.')
                        return
                    
                    # train model and/or evaluate it
                    if train:
                        v_losses.append(training(model, train_loader, val_loader, test_loader, model_type, filename, id))
                    else:
                        evaluation_only(test_loader)
                    
                    if feature_permutation == ETData.NONE:
                        break

                if not test_only_one_subj:
                    break
    
    print('Finished hyperparameter search.')
    #print('Best validation loss: ', min(v_losses), ' with learning rate: ', lrs[v_losses.index(min(v_losses))])


def training(model, train_loader, val_loader, test_loader, model_type, filename, acc_id):

    # if file already exists increase model_nr, without overwriting existing models
    while os.path.exists('saved_models/' + model_type + '/' + filename):
        global model_nr
        model_nr += 1
        #filename = model_type + '_' + str(model_nr) + '.pt'
        if test_only_one_subj:
            filename = model_type + '_subj_' + str(acc_id) + '_' + str(model_nr) + '.pt'
        else:
            filename = model_type + '_' + str(model_nr) + '.pt'

    filepath = 'saved_models/' + model_type + '/'

    if not os.path.exists(filepath):
        os.makedirs(filepath)
    # train model
    losses, vlosses, best_val_loss = model.train(train_loader, val_loader, n_epochs=num_epochs, filename=filename)
    model.save_model(filepath + filename)
    print('Model saved as:', filename)
    
    # evaluate model
    y_test, y_pred = eval.evaluate_model(model, test_loader)

    #eval.plot_losses(losses, vlosses)
    #eval.plot_predictions(y_test, y_pred)

    # save y_test, y_pred and losses
    # create directory if it does not exist
    if test_only_one_subj:
        filepath = 'results/' + model_type + '/' + model_type + '_subj_' + str(acc_id) + '_' + str(model_nr) 
    else:
        filepath = 'results/' + model_type + '/' + model_type + '_' + str(model_nr)

    if not os.path.exists(filepath):
        os.makedirs(filepath)

    np.save(filepath + '/losses.npy', np.array(losses))
    np.save(filepath + '/vlosses.npy', np.array(vlosses))
    np.save(filepath + '/y_test.npy', y_test)
    np.save(filepath + '/y_pred.npy', y_pred)

    #plt.show()
    #input()

    return best_val_loss

def evaluation_only(test_loader):
    global model_type, filename, perm_acc

    model_trainer = Trainer(model_type)
    model = model_trainer.model
    model = model_trainer.load_model('saved_models/' + model_type + '/' + filename)

    if model is None:
        print('Model not found.')
        return
    else:
        print('Model loaded successfully.')
        y_test, y_pred = eval.evaluate_model(model_trainer, test_loader)

        if feature_permutation != ETData.NONE:
            if feature_permutation == ETData.ECCENTRICITY:
                perm_type = 'eccentricity'
            elif feature_permutation == ETData.VERGENCE:
                perm_type = 'vergence'
            elif feature_permutation == ETData.DEPTH:
                perm_type = 'depth'

            model_name = filename.split('.')[0]
            filepath = 'results/' + model_type + '/' + model_name + '_' + perm_type + '_' + str(perm_acc)
            if not os.path.exists(filepath):
                os.makedirs(filepath)

            np.save(filepath + '/y_test.npy', y_test)
            np.save(filepath + '/y_pred.npy', y_pred)
        
            print('Evaluation successful. Results saved in:', filepath)
        else:
            print('Evaluation successful.')


if __name__ == "__main__":
    __main__()