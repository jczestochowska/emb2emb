from torch import nn
import torch
from random import shuffle
from torch.nn.modules.loss import L1Loss
import numpy as np
import os

from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class PerplexityRegressor(nn.Module):
    """
    """

    def __init__(self, input_size, hidden_size, dropout=0., gaussian_noise_std=0.):
        super(PerplexityRegressor, self).__init__()

        self.regressor = nn.Sequential(nn.Dropout(dropout),
                                       nn.Linear(input_size, hidden_size),
                                       nn.ReLU(),
                                    #    nn.Dropout(dropout),
                                    #    nn.Linear(hidden_size, hidden_size // 2),
                                    #    nn.ReLU(),
                                       nn.Dropout(dropout),
                                       nn.Linear(hidden_size, 1))

        self.gaussian_noise_std = gaussian_noise_std

    def forward(self, inputs):

        if self.training and self.gaussian_noise_std > 0.:
            inputs = inputs + \
                torch.randn_like(inputs) * self.gaussian_noise_std

        return self.regressor(inputs)


def freeze(m):
    for p in m.parameters():
        p.requires_grad = False


def train_perplexity_regressor(inputs, encoder, params):

    outputmodelname = params.outputmodelname.split(".")
    outputmodelname = outputmodelname[0] + "_perplexity_regressor." + outputmodelname[1]
    if params.load_perplexity_reg:
        perplexity_regressor = PerplexityRegressor(
            params.embedding_dim, params.embedding_dim // 2, 0., 0.).to(encoder.device)
        checkpoint = torch.load(params.perplexity_regressor_path,
                                map_location=params.device)
        perplexity_regressor.load_state_dict(checkpoint["model_state_dict"])
        return perplexity_regressor

    indices = list(range(len(inputs)))
    inputs = np.array(inputs)
    #num_val_samples = int(0.3 * len(inputs))


    # # calculate perplexity
    # model = GPT2LMHeadModel.from_pretrained('gpt2').to(params.device)
    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # targets = []
    # for input_ in tqdm(inputs):
    #     perplexity = get_perplexity(input_, model, tokenizer, params.device)
    #     # perplexity_scaled = (perplexity - MIN_PERPLEXITY) / (MAX_PERPLEXITY - MIN_PERPLEXITY)
    #     targets.append(perplexity)
    # targets = np.array(targets)

    import pickle
    with open("./data/perplexity/perplexities_log.pkl", 'rb') as f:
        targets = pickle.load(f)
    # import sys
    # sys.exit()

    targets = targets[:10000]
    inputs = inputs[:10000]
    num_val_samples = 3000
    # indices = list(range(len(inputs)))

    # get validation set
    shuffle(indices)
    val_inputs = inputs[indices[-num_val_samples:]]
    val_targets = targets[indices[-num_val_samples:]]
    inputs = inputs[indices[:-num_val_samples]]
    targets = targets[indices[:-num_val_samples]]
    indices = list(range(len(inputs)))

    perplexity_regressor = PerplexityRegressor(params.embedding_dim,
                                               params.embedding_dim // 2,
                                               params.dropout_binary,
                                               params.gaussian_noise_binary).to(encoder.device)
    opt = torch.optim.Adam(perplexity_regressor.parameters(), lr=params.lr_pxtyreg)
    freeze(encoder)
    encoder.eval()
    loss_f = L1Loss()

    def save_reg():
        checkpoint = {"model_state_dict": perplexity_regressor.state_dict()}
        torch.save(checkpoint, os.path.join(params.outputdir, outputmodelname))

    best_mae = evaluate(val_inputs, val_targets, encoder,
                        perplexity_regressor, params)
    bsize = params.batch_size
    error = 0.
    for e in range(params.n_epochs_regressor):

        # shuffle data in each epoch
        shuffle(indices)
        inputs = inputs[indices]
        targets = targets[indices]

        perplexity_regressor.train()
        losses = []
        for idx in range(0, len(inputs), bsize):
            ib = inputs[idx: idx + bsize]
            tb = targets[idx: idx + bsize]

            tb = torch.tensor(tb, device=encoder.device).view(-1, 1).float()
            with torch.no_grad():
                embeddings = encoder(ib)
            preds = perplexity_regressor(embeddings)
            loss = loss_f(preds, tb)
            error += torch.abs(preds - tb).sum().item()

            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

            if (idx / bsize) % params.log_freq == 0:
                avg_loss = np.nanmean(np.array(losses[-params.log_freq:]))
                print("Perplexity regression step {}<->{}: loss/train mae: {} ; val mae: {}".format(e,
                                                                                                    idx,
                                                                                                    avg_loss,
                                                                                                    best_mae))

        val_mae = evaluate(val_inputs, val_targets, encoder,
                           perplexity_regressor, params)
        if val_mae < best_mae:
            best_mae = val_mae
            save_reg()
        print("Loss in epoch {}: {}".format(e, np.nanmean(np.array(losses))))
        error = 0.
    sys.exit()
    return perplexity_regressor


def evaluate(val_inputs, val_targets, encoder, perplexity_regressor, params):
    inputs = val_inputs
    t = val_targets
    bsize = params.batch_size

    errors = []
    perplexity_regressor.eval()

    for idx in range(0, len(inputs), bsize):
        ib = inputs[idx: idx + bsize]
        tb = t[idx: idx + bsize]

        tb = torch.tensor(tb, device=encoder.device).view(-1, 1).float()
        with torch.no_grad():
            embeddings = encoder(ib)
        preds = perplexity_regressor(embeddings)
        errors.append(torch.abs(preds - tb).sum().item())
    return np.nanmean(np.array(errors))


# Sliding window approach
# https://huggingface.co/transformers/perplexity.html
def get_perplexity(sentence, model, tokenizer, device):
    stride = 512
    max_length = model.config.n_positions
    encodings = tokenizer(sentence, return_tensors='pt')
    nlls = []
    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs[0] * trg_len

        nlls.append(neg_log_likelihood)
    cross_entropy = torch.stack(nlls).sum() / end_loc
    # This is also equivalent to the exponentiation of the cross-entropy
    # between the data and model predictions.
    return torch.exp(cross_entropy).item()
