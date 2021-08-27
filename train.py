import torch
import numpy as np
from torch import nn
from model import Net
from tqdm import tqdm
from torch.utils.data import random_split, DataLoader

# hyperparameters
save_model = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_epochs = 5
batch_size = 32
lr = 3e-4
print(device)

# loading the created data samples and mounting them on to the DataLoader
data = torch.load('new_data.pth.tar')
train_data, val_data = random_split(data, [2000, 1000])
train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=2, pin_memory=True)
x, y = next(iter(train_loader))
print(len(data), x.shape, y.shape)

# creating neural net
architecture = [1, 1, 1, 1] # use a bigger model(like [4, 4, 4, 4] or [6, 6, 6, 6]) to improve the accuracy but i can't do it due to CPU/GPU constraints
net = Net(architecture).to(device)

# triplet loss is the loss function which is generally used in one-shot learning face recognition tasks
class TripletLoss(nn.Module):
    '''
    similar to the structure of max-margin loss
    computes the distances b/w the encodings, alpha is the margin(same as the margin used in max-margin loss)
    '''
    def __init__(self, alpha=10):
        super().__init__()
        self.alpha = alpha

    def forward(self, anchor, pos, neg):
        pos_dist = torch.sum((anchor - pos)**2, dim=-1)
        neg_dist = torch.sum((anchor - neg)**2, dim=-1)
        basic_loss = pos_dist - neg_dist + self.alpha
        zeros = torch.zeros_like(basic_loss)
        return torch.max(basic_loss, zeros).sum()

# using adam optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
loss_fn = TripletLoss()

# creating the train/validation loop
def loop(net, loader, is_train):
    net.train(is_train)
    losses = []
    pbar = tqdm(loader, total=len(loader))
    for x, y in pbar:
        x = x.to(device)
        y0 = y[:, 0].to(device)
        y1 = y[:, 1].to(device)
        with torch.set_grad_enabled(is_train):
            anchor = net(x) # getting the anchor encoding
            pos = net(y0) # getting the positive image encoding
            neg = net(y1) # getting the negative image encoding
            loss = loss_fn(anchor, pos, neg)
            losses.append(loss.item())
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        pbar.set_description(f'epoch={epoch}, train={int(is_train)}, loss={np.mean(losses):.4f}')

# train and validation loop
for epoch in range(n_epochs):
    loop(net, train_loader, True)
    loop(net, val_loader, False)

# saving the model
if save_model:
    torch.save(net.state_dict(), 'trained_model.pth.tar')