from prediction.decode import decode
import torch
from torch.nn.functional import ctc_loss, log_softmax
from rapidfuzz.distance import Levenshtein

import numpy as np 

from tqdm import tqdm
import wandb
import gc


def eval_model(model, config, val_dataloader):
    model.eval()
    val_losses = []
    epoch_metrics = []
    prediciton = []
    for i, b in enumerate(tqdm(val_dataloader, total=len(val_dataloader))):
        images = b["image"].to(config['device'])
        seqs_gt = b["seq"]
        seq_lens_gt = b["seq_len"]

        with torch.no_grad():
            seqs_pred = model(images).cpu()

        seqs_decoded = decode(seqs_pred)
        epoch_metrics.append(Levenshtein.distance(seqs_decoded, b["text"]))

        if config['return_preds']:
            prediciton.append(seqs_decoded)
        
        log_probs = log_softmax(seqs_pred, dim=2)
        seq_lens_pred = torch.Tensor([seqs_pred.size(0)] * seqs_pred.size(1)).int()

        loss = ctc_loss(log_probs=log_probs,          # (T, N, C)
                        targets=seqs_gt,              # N, S or sum(target_lengths)
                        input_lengths=seq_lens_pred,  # N
                        target_lengths=seq_lens_gt)   # N

        val_losses.append(loss.item())

        # clean tresh
        del b
        gc.collect()
        torch.cuda.empty_cache()

    val_loss = np.mean(val_losses)
    val_metrics = np.mean(epoch_metrics)

    print('VALIDATION Loss: ', val_loss, 'metrics: ', val_metrics)
    wandb.log({"VALIDATION loss": val_loss, "VALIDATION metrics": val_metrics}) 


    if config['return_preds']:
        return prediciton

if __name__ == '__main__':
    print('Everything is ready!')
