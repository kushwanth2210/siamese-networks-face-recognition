'''
to use a siamese network, a dataset with 3 images as one sample is required
out of these 3 images, 2 images are of the same person, 1 image is of a different person
2 images of the same person are called anchor and positive image, 1 image of different person is called a negative image
'''

import torch
import random
import numpy as np
import torchvision.transforms as T
from torch.utils.data import TensorDataset
from torchvision.datasets import ImageFolder

n_data_samples = 3000 # try using 10000 here
transform = T.Compose(
                [
                    T.Resize((224, 224)),
                    T.ToTensor()
                ]
            )

data = ImageFolder('../new_dataset', transform=transform)
print(len(data), len(data.classes))
labels = [data[i][1] for i in range(len(data))]

def get_sample(data, labels):
    '''
    iterates thorugh trainset directory and creates a new data sample with
    3 images of the format mentioned above
    '''
    curr_idxs = []
    while curr_idxs == []:
        curr_label = np.random.randint(max(labels))
        curr_idxs = [i for i, x in enumerate(labels) if x == curr_label]

    if len(curr_idxs) > 1:
        anc_pos = random.sample(curr_idxs, 2)
        anchor, pos = data[anc_pos[0]][0], data[anc_pos[1]][0]
    else:
        anc_pos = curr_idxs
        anchor, pos = data[anc_pos[0]][0], data[anc_pos[0]][0]

    neg_idx = anc_pos[0]
    while neg_idx in curr_idxs:
        neg_idx = np.random.randint(len(labels))

    neg = data[neg_idx][0]
    tgt = torch.stack([pos, neg])
    return anchor, tgt

def create_data_samples(data, labels, n_data_samples=1000):
    '''
    takes the data samples created by the get_sample() function and creates
    a TensorDataset instance to work with pytorch DataLoader
    '''
    x = torch.zeros(n_data_samples, 3, 224, 224)
    y = torch.zeros(n_data_samples, 2, 3, 224, 224)
    for i in range(n_data_samples):
        print(i)
        anchor, tgt = get_sample(data, labels)
        x[i] = anchor
        y[i] = tgt
    data = TensorDataset(x, y)
    return data

# creating data samples and instantiating the TensorDataset object and saving it to working dir
# i created 3000 data samples(3 images per each sample) and i can't create more samples due to CPU/GPU constraints, but feel free to create 10000 data samples if possible
# 10000 data samples is a ideal amount to train a siamese network
print('creating data samples:')
data = create_data_samples(data, labels, n_data_samples=n_data_samples)
torch.save(data, 'new_data.pth.tar')
print('done!!')