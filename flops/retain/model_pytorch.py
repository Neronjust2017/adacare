""" Matplotlib backend configuration """
import matplotlib

matplotlib.use('PS')  # generate postscript output by default

""" Imports """
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import numpy as np
from scipy.sparse import coo_matrix
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

import sys
import argparse
import pickle
import time
from tqdm import tnrange, tqdm_notebook
from thop import profile

""" Arguments """
parser = argparse.ArgumentParser()

parser.add_argument('data_path', metavar='DATA_PATH', help="Path to the dataset")
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, metavar='LR',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--weight-decay', '--wd', default=0, type=float, metavar='W', help='weight decay (default: 0)')
parser.add_argument('--epochs', default=10, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N',
                    help='mini-batch size for train (default: 64)')
parser.add_argument('--eval-batch-size', type=int, default=256, help='mini-batch size for eval (default: 1000)')

parser.add_argument('--no-cuda', dest='cuda', action='store_false', help='NOT use cuda')
parser.add_argument('--no-plot', dest='plot', action='store_false', help='no plot')
parser.add_argument('--threads', type=int, default=-1,
                    help='number of threads for data loader to use (default: -1 = (multiprocessing.cpu_count()-1 or 1))')
parser.add_argument('--seed', type=int, default=0, help='random seed to use. Default=0')

parser.add_argument('--save', default='./', type=str, metavar='SAVE_PATH',
                    help='path to save checkpoints (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='LOAD_PATH',
                    help='path to latest checkpoint (default: none)')

parser.set_defaults(cuda=True, plot=True)

""" Helper Functions """


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


""" Custom Dataset """


class VisitSequenceWithLabelDataset(Dataset):
    def __init__(self, seqs, labels, num_features, reverse=True):
        """
		Args:
			seqs (list): list of patients (list) of visits (list) of codes (int) that contains visit sequences
			labels (list): list of labels (int)
			num_features (int): number of total features available
			reverse (bool): If true, reverse the order of sequence (for RETAIN)
		"""

        if len(seqs) != len(labels):
            raise ValueError("Seqs and Labels have different lengths")

        self.seqs = []
        # self.labels = []

        for seq, label in zip(seqs, labels):

            if reverse:
                sequence = list(reversed(seq))
            else:
                sequence = seq

            row = []
            col = []
            val = []
            for i, visit in enumerate(sequence):
                for code in visit:
                    if code < num_features:
                        row.append(i)
                        col.append(code)
                        val.append(1.0)

            self.seqs.append(coo_matrix((np.array(val, dtype=np.float32), (np.array(row), np.array(col))),
                                        shape=(len(sequence), num_features)))
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.seqs[index], self.labels[index]


""" Custom collate_fn for DataLoader"""


# @profile
def visit_collate_fn(batch):
    """
	DataLoaderIter call - self.collate_fn([self.dataset[i] for i in indices])
	Thus, 'batch' is a list [(seq1, label1), (seq2, label2), ... , (seqN, labelN)]
	where N is minibatch size, seq is a SparseFloatTensor, and label is a LongTensor
	:returns
		seqs
		labels
		lengths
	"""
    batch_seq, batch_label = zip(*batch)

    num_features = batch_seq[0].shape[1]
    seq_lengths = list(map(lambda patient_tensor: patient_tensor.shape[0], batch_seq))
    max_length = max(seq_lengths)

    sorted_indices, sorted_lengths = zip(*sorted(enumerate(seq_lengths), key=lambda x: x[1], reverse=True))
    sorted_padded_seqs = []
    sorted_labels = []

    for i in sorted_indices:
        length = batch_seq[i].shape[0]

        if length < max_length:
            padded = np.concatenate(
                (batch_seq[i].toarray(), np.zeros((max_length - length, num_features), dtype=np.float32)), axis=0)
        else:
            padded = batch_seq[i].toarray()

        sorted_padded_seqs.append(padded)
        sorted_labels.append(batch_label[i])

    seq_tensor = np.stack(sorted_padded_seqs, axis=0)
    label_tensor = torch.LongTensor(sorted_labels)

    return torch.from_numpy(seq_tensor), label_tensor, list(sorted_lengths)


""" RETAIN model class """


class RETAIN(nn.Module):
    def __init__(self, dim_input, dim_emb=128, dropout_input=0.8, dropout_emb=0.5, dim_alpha=128, dim_beta=128,
                 dropout_context=0.5, dim_output=2, l2=0.0001, batch_first=True):
        super(RETAIN, self).__init__()
        self.batch_first = batch_first
        self.embedding = nn.Sequential(
            nn.Dropout(p=dropout_input),
            nn.Linear(dim_input, dim_emb, bias=False),
            nn.Dropout(p=dropout_emb)
        )
        init.xavier_normal(self.embedding[1].weight)

        self.rnn_alpha = nn.GRU(input_size=dim_emb, hidden_size=dim_alpha, num_layers=1, batch_first=self.batch_first)

        self.alpha_fc = nn.Linear(in_features=dim_alpha, out_features=1)
        init.xavier_normal(self.alpha_fc.weight)
        self.alpha_fc.bias.data.zero_()

        self.rnn_beta = nn.GRU(input_size=dim_emb, hidden_size=dim_beta, num_layers=1, batch_first=self.batch_first)

        self.beta_fc = nn.Linear(in_features=dim_beta, out_features=dim_emb)
        init.xavier_normal(self.beta_fc.weight, gain=nn.init.calculate_gain('tanh'))
        self.beta_fc.bias.data.zero_()

        self.output = nn.Sequential(
            nn.Dropout(p=dropout_context),
            nn.Linear(in_features=dim_emb, out_features=dim_output)
        )
        init.xavier_normal(self.output[1].weight)
        self.output[1].bias.data.zero_()

    def forward(self, x, lengths):
        if self.batch_first:
            batch_size, max_len = x.size()[:2]
        else:
            max_len, batch_size = x.size()[:2]

        # emb -> batch_size X max_len X dim_emb
        emb = self.embedding(x)

        packed_input = pack_padded_sequence(emb, lengths, batch_first=self.batch_first)

        g, _ = self.rnn_alpha(packed_input)

        # alpha_unpacked -> batch_size X max_len X dim_alpha
        alpha_unpacked, _ = pad_packed_sequence(g, batch_first=self.batch_first)

        # mask -> batch_size X max_len X 1
        mask = Variable(torch.FloatTensor(
            [[1.0 if i < lengths[idx] else 0.0 for i in range(max_len)] for idx in range(batch_size)]).unsqueeze(2),
                        requires_grad=False)
        if next(self.parameters()).is_cuda:  # returns a boolean
            mask = mask.cuda()

        # e => batch_size X max_len X 1
        e = self.alpha_fc(alpha_unpacked)

        def masked_softmax(batch_tensor, mask):
            exp = torch.exp(batch_tensor)
            masked_exp = exp * mask
            sum_masked_exp = torch.sum(masked_exp, dim=1, keepdim=True)
            return masked_exp / sum_masked_exp

        # Alpha = batch_size X max_len X 1
        # alpha value for padded visits (zero) will be zero
        alpha = masked_softmax(e, mask)

        h, _ = self.rnn_beta(packed_input)

        # beta_unpacked -> batch_size X max_len X dim_beta
        beta_unpacked, _ = pad_packed_sequence(h, batch_first=self.batch_first)

        # Beta -> batch_size X max_len X dim_emb
        # beta for padded visits will be zero-vectors
        beta = F.tanh(self.beta_fc(beta_unpacked) * mask)

        # context -> batch_size X (1) X dim_emb (squeezed)
        # Context up to i-th visit context_i = sum(alpha_j * beta_j * emb_j)
        # Vectorized sum
        context = torch.bmm(torch.transpose(alpha, 1, 2), beta * emb).squeeze(1)

        # without applying non-linearity
        logit = self.output(context)

        return logit, alpha, beta


""" Epoch function """


def epoch(loader, model, criterion, optimizer=None, train=False):
    if train and not optimizer:
        raise AttributeError("Optimizer should be given for training")

    if train:
        model.train()
        mode = 'Train'
    else:
        model.eval()
        mode = 'Eval'

    losses = AverageMeter()
    labels = []
    outputs = []

    for bi, batch in enumerate(tqdm_notebook(loader, desc="{} batches".format(mode), leave=False)):
        inputs, targets, lengths = batch

        input_var = torch.autograd.Variable(inputs)
        target_var = torch.autograd.Variable(targets)
        if args.cuda:
            input_var = input_var.cuda()
            target_var = target_var.cuda()

        output, alpha, beta = model(input_var, lengths)
        loss = criterion(output, target_var)
        assert not np.isnan(loss.data[0]), 'Model diverged with loss = NaN'

        labels.append(targets)

        # since the outputs are logit, not probabilities
        outputs.append(F.softmax(output).data)

        # record loss
        losses.update(loss.data[0], inputs.size(0))

        # compute gradient and do update step
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return torch.cat(labels, 0), torch.cat(outputs, 0), losses.avg


""" Main function """

if __name__ == "__main__":

    print('Constructing model ... ')
    # device = torch.device("cuda:0" if torch.cuda.is_available() == True else 'cpu')
    device = 'cpu'
    print("available device: {}".format(device))
    batch_x = torch.rand(128, 400, 76)
    batch_x = torch.tensor(batch_x, dtype=torch.float32).to(device)

    batch_lengths = torch.ones(128) * 400
    batch_lengths = torch.tensor(batch_lengths, dtype=torch.float32).to(device)

    model = RETAIN(dim_input=76,
                   dim_emb=512,
                   dropout_emb=0.5,
                   dim_alpha=512,
                   dim_beta=512,
                   dropout_context=0.5,
                   dim_output=2)

    _, _, _ = model(batch_x, batch_lengths)
    flops, params = profile(model, inputs=[batch_x, batch_lengths])
    print("%.2fG" % (flops / 1e9), "%.2fM" % (params / 1e6))
    print('!!!!!!!')

    from torchsummaryX import summary
    summary(model, batch_x, batch_lengths)

    # from torchstat import stat
    # stat(model, [(128, 400, 76), (128)])

    # from ptflops import get_model_complexity_info
    # macs, params = get_model_complexity_info(model, [(128, 400, 76), (128)], as_strings=True,
    #                                          print_per_layer_stat=True, verbose=True)
