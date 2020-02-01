import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from dataset import HandwritingDataSet, pad_collate
from model import SynthesisNetwork
from torch.utils.data import DataLoader
from torch.autograd import Variable
from loss import NLLoss

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS = 10
BATCH_SIZE = 16

if __name__ == '__main__':
    train_set = HandwritingDataSet()
    test_set = HandwritingDataSet(train=False)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, collate_fn=pad_collate, drop_last=True)
    test_loader = DataLoader(train_set, batch_size=1, collate_fn=pad_collate, drop_last=True)


    model = SynthesisNetwork(
        vocab_size = train_loader.dataset.vocab_size,
        hidden_size = 400,
        batch_size = BATCH_SIZE,
    )
    opt = torch.optim.Adam(model.parameters(), lr=0.0005)



    for epoch_i in range(1, NUM_EPOCHS+1):
        epoch_losses = []
        for i, batch in enumerate(train_loader):
            prev_states = None
            sentences, strokes, sentences_oh, sent_masks, stroke_masks = batch
            stroke_masks = Variable(stroke_masks, requires_grad=False)
            model_outs = model(batch, epoch_i, prev_states)
            attention_out = model_outs[0]
            e_hat, pi_hat, mu1_hat, mu2_hat, sigma1_hat, sigma2_hat, rho_hat = model_outs[1]
            h_1t, w_t, k_t, h_2t = model_outs[2]

            # Loss
            nll = NLLoss(model_outs[1], strokes, stroke_masks)
            loss = nll.get_loss()
            print(loss)
            epoch_losses.append(loss.item())
            opt.zero_grad()
            loss.backward()
            for name, p in model.named_parameters():
                if 'lstm' in name:
                    p.grad.data.clamp_(-10, 10)
                elif 'linear' in name:
                    p.grad.data.clamp_(-100, 100)
            opt.step()
        print('End of epoch {}'.format(epoch_i))
            # avg_loss = sum(epoch_losses) / len(epoch_losses)
            # print('Epoch {} : \n \t train_loss: {}'.format(epoch_i, avg_loss))
