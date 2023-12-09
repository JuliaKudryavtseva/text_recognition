from utils.decode import decode
from utils.validation import eval_model
import torch
from torch.nn.functional import ctc_loss, log_softmax
from rapidfuzz.distance import Levenshtein

import numpy as np 

from tqdm import tqdm
import time
import wandb
import gc
import os


def training(model, config, train_dataloader, val_dataloader, optimizer, log=True):
    
    epoch=0
    if config['checkpoint']:
        model, optimizer,  epoch = load_checkpoint(model, config['save_path'], optimizer)


    best_loss = np.inf
    best_metrics = np.inf

    for i, epoch in enumerate(range(config['num_epochs']), epoch):
        model.train()
        start_time = time.time()

        epoch_losses = []
        epoch_metrics = []

        for j, b in enumerate(tqdm(train_dataloader, total=len(train_dataloader))):
            images = b["image"].to(config['device'])
            seqs_gt = b["seq"]
            seq_lens_gt = b["seq_len"]

            seqs_pred = model(images).cpu()

            seqs_decoded = decode(seqs_pred)
            epoch_metrics.append(
                Levenshtein.distance(seqs_decoded, b["text"]))


            log_probs = log_softmax(seqs_pred, dim=2)
            seq_lens_pred = torch.Tensor([seqs_pred.size(0)] * seqs_pred.size(1)).int()

            loss = ctc_loss(log_probs=log_probs,          # (T, N, C)
                            targets=seqs_gt,              # N, S or sum(target_lengths)
                            input_lengths=seq_lens_pred,  # N
                            target_lengths=seq_lens_gt)   # N

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            # clean tresh
            del b
            gc.collect()
            torch.cuda.empty_cache() 

        epoch_loss = round(np.mean(epoch_losses), 3)
        epoch_metrics  = round(np.mean(epoch_metrics), 3)
        
        if epoch_metrics < best_metrics:
            best_metrics = epoch_metrics

            name = config['name']
            name_checkpoint = f'{name}.pth'
            save_path = os.path.join(os.getcwd(), config['save_path'], name_checkpoint)
            save_checkpoint(model, optimizer, save_path, epoch)

        end_time = time.time()
        epoch_time=round(end_time-start_time, 3)
        
        print(f'TRAIN: {epoch_loss=}, {epoch_metrics=}, time: {epoch_time}')
        eval_model(model, config, val_dataloader, test=False, log=log)
        if log:
            wandb.log({"TRAIN loss": epoch_loss, "TRAIN metrics": epoch_metrics})


    

def save_checkpoint(model, optimizer, filename, EPOCH):
        with open(filename, "wb") as fp:
            torch.save(model.state_dict(), fp)

            torch.save({
                'epoch': EPOCH,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, filename)
            

def load_checkpoint(model, filename, optimizer=None):
    loaded_checkpoint = {}

    with open(filename, "rb") as fp:
        checkpoint = torch.load(fp, map_location="cpu")
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer:
             optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
             
    print('ckeckpoint loaded')
    return model, optimizer,  checkpoint['epoch']

if __name__ == '__main__':    
    print('Everything is ready!')
