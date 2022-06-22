import os
import torch
import logging

from torch.serialization import default_restore_location

def save_checkpoint(config, model, optimizer, epoch, valid_loss):
    os.makedirs(config['save_dir'], exist_ok=True)
    last_epoch = getattr(save_checkpoint, 'last_epoch', -1)
    save_checkpoint.last_epoch = max(last_epoch, epoch)
    prev_best = getattr(save_checkpoint, 'best_loss', float('inf'))
    save_checkpoint.best_loss = min(prev_best, valid_loss)

    state_dict = {
        'epoch': epoch,
        'val_loss': valid_loss,
        'best_loss': save_checkpoint.best_loss,
        'last_epoch': save_checkpoint.last_epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'args': config,
    }

    if valid_loss < prev_best:
        torch.save(state_dict, os.path.join(config['save_dir'], 'checkpoint_best.pt'))
    if last_epoch < epoch:
        torch.save(state_dict, os.path.join(config['save_dir'], 'checkpoint_last.pt'))




def load_checkpoint(config, model, optimizer):
    checkpoint_path = os.path.join(config['save_dir'], config['restore_file'])
    if os.path.isfile(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=lambda s, l: default_restore_location(s, 'cpu'))
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        save_checkpoint.best_loss = state_dict['best_loss']
        save_checkpoint.last_epoch = state_dict['last_epoch']
        logging.info('Loaded checkpoint {}'.format(checkpoint_path))
        return state_dict
