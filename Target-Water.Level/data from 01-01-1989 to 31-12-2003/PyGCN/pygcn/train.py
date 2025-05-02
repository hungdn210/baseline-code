from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from pygcn.utils import load_data
from pygcn.models import GCN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()

# Convert to PyTorch tensors if necessary
if isinstance(features, np.ndarray):
    features = torch.FloatTensor(features)
if isinstance(labels, np.ndarray):
    labels = torch.FloatTensor(labels)

# Normalize labels using StandardScaler
scaler = StandardScaler()
labels_np = labels.numpy().reshape(-1, 1)
labels_scaled = scaler.fit_transform(labels_np).flatten()
labels = torch.tensor(labels_scaled, dtype=torch.float32)

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj).view(-1)
    criterion = torch.nn.MSELoss()
    loss_train = criterion(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        model.eval()
        output = model(features, adj).view(-1)

    loss_val = criterion(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

def test():
    model.eval()
    output = model(features, adj).view(-1)
    output_test = output[idx_test].detach().cpu().numpy()
    y_true = labels[idx_test].cpu().numpy()

    # Denormalize both
    output_denorm = scaler.inverse_transform(output_test.reshape(-1, 1)).flatten()
    y_true_denorm = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()

    mae = mean_absolute_error(y_true_denorm, output_denorm)
    rmse = mean_squared_error(y_true_denorm, output_denorm, squared=False)

    print("Test set results:")
    print("MAE: {:.4f}".format(mae))
    print("RMSE: {:.4f}".format(rmse))


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
