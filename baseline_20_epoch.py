from utils.datasets.dataset import get_dataloaders
from model.model import CRNN
from utils.train_model import training, load_checkpoint
from utils.validation import eval_model

import torch
import wandb
import os

import warnings
warnings.filterwarnings('ignore')


config = {
    'name'      : 'baseline',


    'num_epochs': 20,
    'batch_size': 16,
    'device'    : 'cuda:0',

    'checkpoint': False,
    'save_path' : 'checkpoints',
    'log'       : True,

    'weight_decay': 1e-3
}



if __name__ == '__main__':

    if config['log']:
        wandb.login(key="cfedead01b64744b86b1cc0779b5ab7c10fc942f")

        wandb.init(
            project="ML3_HW3",
            name=config['name'],
            reinit=True,
            config=config,
        )

    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(BATCH_SIZE=config['batch_size'], train_ratio=0.9)

    model = CRNN()
    model.to(config['device'])

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, amsgrad=True, weight_decay=config['weight_decay'])

    training(model, config, train_dataloader, val_dataloader, optimizer, config['log'])

    # make predictions
    del model
    model = CRNN()
    name = config['name']
    load_checkpoint(model, os.path.join(config['save_path'], f'{name}.pth'))
    model.to(config['device'])
    eval_model(model, config, test_dataloader, test=True, log=False)

    if config['log']:
        wandb.finish()

    print('FINISH')
