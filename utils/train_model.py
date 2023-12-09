
from prediction.decode import decode
import torch
from torch.nn.functional import ctc_loss, log_softmax
from rapidfuzz.distance import Levenshtein

import numpy as np 

from tqdm import tqdm
import wandb
import gc


def training(model, config, train_dataloader, optimizer):
    
    epoch=0
    if config['checkpoint']:
        model, optimizer,  epoch = load_checkpoint(model, config['save_path'], optimizer)


    best_loss = np.inf
    best_metrics = np.inf

    model.train()
    for i, epoch in enumerate(range(config['num_epochs']), epoch):
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

        avr_epoch_loss = np.mean(epoch_losses)
        avr_epoch_metrics  = np.mean(epoch_metrics)
        
        if avr_epoch_metrics < best_metrics:
            best_metrics = avr_epoch_metrics
            save_checkpoint(model, optimizer, config['save_path'], epoch)

        
        print(f'TRAIN: {avr_epoch_loss=}, {avr_epoch_metrics=}')
        wandb.log({"TRAIN loss": avr_epoch_loss, "TRAIN metrics": avr_epoch_metrics})


    

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
