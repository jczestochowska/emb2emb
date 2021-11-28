import numpy as np

import torch
from torch.autograd import Variable

import usc_dae_utils.operations as operations

# Inspired by autoencode.py from usc_dae model




def additive_noise(sent_batch,
                   lengths,
                   next_batch,
                   ae_add_noise_perc_per_sent_low = 0.2,
                   ae_add_noise_perc_per_sent_high = 0.5,
                   ae_add_noise_num_sent = 2,
                   ae_add_noise_2_grams = True):
    """
    Default values taken from usc_dae config
    "ae_add_noise_perc_per_sent_low": 0.2,
    "ae_add_noise_perc_per_sent_high": 0.5,
    "ae_add_noise_num_sent": 2,
    "ae_add_noise_2_grams": true,
    """
    assert ae_add_noise_perc_per_sent_low <= ae_add_noise_perc_per_sent_high
    batch_size = len(lengths)
    if ae_add_noise_2_grams:
        shuffler_func = operations.shuffle_2_grams
    else:
        shuffler_func = operations.shuffle
    split_sent_batch = [
        sent.split()
        for sent in sent_batch
    ]
    length_arr = np.array(lengths)
    min_add_lengths = np.floor(length_arr * ae_add_noise_perc_per_sent_low)
    max_add_lengths = np.ceil(length_arr * ae_add_noise_perc_per_sent_high)
    for s_i in range(ae_add_noise_num_sent):
        add_lengths = np.round(
            np.random.uniform(min_add_lengths, max_add_lengths)
        ).astype(int)
        next_batch = operations.shuffle(next_batch)
        for r_i, new_sent in enumerate(next_batch):
            addition = shuffler_func(new_sent.split())[:add_lengths[r_i]]
            split_sent_batch[r_i] += addition
    noised_sent_batch = [
        " ".join(shuffler_func(sent))
        for sent in split_sent_batch
    ]
    return noised_sent_batch