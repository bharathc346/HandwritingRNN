import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from loss import NLLoss


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class WindowAttention(nn.Module):
    def __init__(self, fc_in_size, fc_out_size):
        super(WindowAttention, self).__init__()
        self.linear = nn.Linear(fc_in_size, fc_out_size)

    def forward(self, h1_out, sentences_oh, kappa_prev, sent_masks, stroke_masks):
        batch_size = sentences_oh.shape[0]
        U = sentences_oh.shape[1] # max sentence length

        # size 3K vector p determined by the outputs of the first hidden layer
        p = self.linear(h1_out)

        # output_size is 3K, split output into 3 for eq 48.
        a_hat, b_hat, kap_hat = p.chunk(3, dim=1) # chunk on 3K dimension
        alpha = a_hat.exp()
        beta = b_hat.exp()
        kappa = kappa_prev + kap_hat.exp()

        # equations 46 and 47s
        # u: B x max_sent_len, from 0 to U in batch
        with torch.no_grad():
            # 1 indexing not to zero out
            u = torch.arange(1, U+1).type(torch.FloatTensor).to(DEVICE).repeat(batch_size,1)
            u = u.unsqueeze(1).repeat(1, kappa.shape[1], 1) # u: B x K x max_sent_len

        # kappa: B x K --> mod_kappa: B x K x max_sent_len
        mod_kappa = kappa.unsqueeze(-1).repeat(1, 1, U)
        sub_square = (mod_kappa - u) ** 2
        exp_term = (-beta.unsqueeze(2) * sub_square).exp()
        phi = (alpha.unsqueeze(2) * exp_term).sum(dim=1) # sum over K
        phi *= sent_masks # apply sent mask to only consider u <= U
        w_t = (phi.unsqueeze(-1) * sentences_oh).sum(dim=1)
        return kappa, w_t


class SynthesisNetwork(nn.Module):
    def __init__(self, vocab_size, hidden_size, batch_size, input_size=3, K=20):
        super(SynthesisNetwork, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.input_size = input_size
        self.K = K
        self.first_input = True

        # Note we can't use stacked lstms here as we won't be able to access
        # the intermediate hidden states, thus we declare 2 lstms
        self.lstm_cell = nn.LSTMCell(
            input_size=self.input_size + self.vocab_size, # concat w_prev and x_t
            hidden_size=self.hidden_size, # h_1prev
        )
        self.lstm2 = nn.LSTM(
            input_size=self.input_size + self.vocab_size + self.hidden_size, # concat w_prev, x_t, and h_1t
            hidden_size=self.hidden_size, # h_2prev
            batch_first=True
        )
        self.win_att = WindowAttention(self.hidden_size, 3 * self.K)
        self.linear = nn.Linear(self.hidden_size * 2, 6 * self.K + 1)

    def forward(self, batch, epoch_i, prev_states=None):
        if epoch_i == 2:
            import pdb; pdb.set_trace()
        sentences, strokes, sentences_oh, sent_masks, stroke_masks = batch

        if not prev_states:
            h_1prev, w_prev, kappa_prev, h_2prev = self.init_states()
        else:
            h_1prev, w_prev, kappa_prev, h_2prev = prev_states

        T = strokes.shape[1]

        # Note we must iterate by time here as we need the
        # Get all time dependent variables
        # u = 0
        lstm_0_out = []
        attention_out = []

        for t in range(T):
            x_t = strokes[:,t,:]
            rnn1_in = torch.cat((x_t, w_prev), dim=1)
            rnn1_out, h_1t = self.lstm_cell(rnn1_in, (h_1prev, h_1prev)) # use h_1prev as cell state

            # window attention
            k_t, w_t = self.win_att(rnn1_out, sentences_oh, kappa_prev, sent_masks, stroke_masks)
            lstm_0_out.append(h_1t)
            attention_out.append(w_t)

        lstm_0_out = torch.stack(lstm_0_out, dim=1) # B x T x H
        attention_out = torch.stack(attention_out, dim=1) # B x T x V
        # second rnn
        rnn2_in = torch.cat((strokes, attention_out, lstm_0_out), dim=-1) # B x T x input_size
        rnn2_out, h_2t = self.lstm2(rnn2_in, h_2prev)

        lin_in = torch.cat([lstm_0_out, rnn2_out], dim=-1)

        # linear layer
        lin_out = self.linear(lin_in) # torch.Size([1, T,  6 * 20 + 1])
        pi_hat = lin_out[:, :, : self.K]
        mu1_hat, mu2_hat = lin_out[:, :, self.K : 2 * self.K], lin_out[:, :, 2 * self.K : 3 * self.K]
        sigma1_hat, sigma2_hat = lin_out[:, :, 3 * self.K :  4 * self.K], lin_out[:, :, 4 * self.K :  5 * self.K]
        rho_hat = lin_out[:, :,  5 * self.K : 6 * self.K]
        e_hat = lin_out[:, :, 6 * self.K]


        # log_pi = torch.log_softmax(lin_out[:, :, : self.K], dim=-1)
        # mu = lin_out[:, :, self.K : 3 * self.K]
        # log_sigma = lin_out[:, :, 3 * self.K : 5 * self.K]
        # rho = torch.tanh(lin_out[:, :, 5 * self.K : 6 * self.K])
        # e = torch.sigmoid(lin_out[:, :, 6 * self.K])
        if torch.isnan(e_hat).any():
            import pdb; pdb.set_trace()

        return (
            attention_out,
            (e_hat, pi_hat, mu1_hat, mu2_hat, sigma1_hat, sigma2_hat, rho_hat),
            (h_1t, w_t, k_t, h_2t)
        )

    def init_states(self):
        w_prev = torch.zeros(self.batch_size, self.vocab_size).to(DEVICE)
        h_1prev = torch.zeros(self.batch_size, self.hidden_size).to(DEVICE)
        # h_2prev = torch.zeros(self.batch_size, self.hidden_size)
        h_2prev = None # this is a bit weird but above kept erroring out
        kappa_prev = torch.zeros(self.batch_size, self.K).to(DEVICE)

        return h_1prev, w_prev, kappa_prev, h_2prev


if __name__ == '__main__':
    batch_size, vocab_size, U, T, K = 1, 58, 39, 1049, 20
    sents = torch.IntTensor(batch_size, U).random_(0, vocab_size)
    stroke_last = torch.FloatTensor(batch_size, T, 1).random_(0,2)
    stroke_first = torch.rand(batch_size, T, 2)
    strokes = torch.cat((stroke_first, stroke_last), dim=-1)
    oh_sents = torch.IntTensor(batch_size, U, vocab_size).random_(0,2)
    sent_masks = torch.IntTensor(batch_size, U).random_(0,vocab_size)
    stroke_masks = torch.IntTensor(batch_size, T, vocab_size).random_(0,2)
    batch = (sents, strokes, oh_sents, sent_masks, stroke_masks)

    ut_model = SynthesisNetwork(
        vocab_size = vocab_size,
        hidden_size = 400,
        batch_size = batch_size,
        input_size = 3,
        K = K
    )
    model_outs = ut_model(batch)
    attention_out = model_outs[0]
    e_hat, pi_hat, mu1_hat, mu2_hat, sigma1_hat, sigma2_hat, rho_hat = model_outs[1]
    h_1t, w_t, k_t, h_2t = model_outs[2]

    assert attention_out.shape[0] == batch_size and attention_out.shape[1] == T and attention_out.shape[2] == vocab_size

    assert e_hat.shape[0] == batch_size and e_hat.shape[-1] == T
    assert pi_hat.shape[0] == batch_size and pi_hat.shape[1] == T and pi_hat.shape[2] == K
    assert mu1_hat.shape[0] == batch_size and mu1_hat.shape[1] == T and mu1_hat.shape[2] == K
    assert mu2_hat.shape[0] == batch_size and mu2_hat.shape[1] == T and mu2_hat.shape[2] == K
    assert sigma1_hat.shape[0] == batch_size and sigma1_hat.shape[1] == T and sigma1_hat.shape[2] == K
    assert sigma2_hat.shape[0] == batch_size and sigma2_hat.shape[1] == T and sigma2_hat.shape[2] == K
    assert rho_hat.shape[0] == batch_size and rho_hat.shape[1] == T and rho_hat.shape[2] == K
