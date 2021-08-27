'''
computes the accuracy on the dataset
'''

import torch
import numpy as np
from torch import nn
from model import Net
from tqdm import trange
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

# creating neural net and loading the trained parameters
build_database = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
architecture = [1, 1, 1, 1]
net = Net(architecture).to(device)
net.load_state_dict(torch.load('trained_model.pth.tar', map_location=device))
net.eval()

transform = T.Compose(
                        [
                            T.Resize((224, 224)),
                            T.ToTensor()
                        ]
                    )
data = ImageFolder('../new_dataset', transform=transform)
print(len(data), len(data.classes))

# database(type: dict) consists of embeddings of all the faces present in the data
if build_database:
    labels = [data[i][1] for i in range(len(data))]
    database = {}
    print('building database...')
    # iteraing over all the faces and getting their embedding by passing them through the neural net and saving them in the database
    for label in trange(len(data.classes)):
        try:
            label_idxs = [i for i, x in enumerate(labels) if x == label]
            img, _ = data[label_idxs[0]]
            img = img.unsqueeze(0).to(device)
            with torch.no_grad():
                enc = net(img)
            database[data.classes[label]] = enc.cpu()
        except IndexError:
            print('')
            print(f'label not found - {label}')

    torch.save(database, f'database_trained_model.pth.tar')
else:
    database = torch.load(f'database_trained_model.pth.tar')

@torch.no_grad()
def get_identity(img):
    '''
    takes an image and outputs their identity, here identities are the labels of the faces present in the dataset
    '''
    enc = net(img)
    min_dist = 1000
    for name, db_enc in database.items():
        dist = torch.linalg.norm(db_enc - enc)
        if dist < min_dist:
            min_dist = dist
            identity = name
    return identity

total = 0
for i in trange(len(data)):
    img, label = data[i]
    img = img.unsqueeze(0).to(device)
    pred = get_identity(img)
    if pred == data.classes[label]:
        total += 1
print(total / len(data)) # accuracy achieved: 26.11%, note that the accuracy will improve if we use more data samples(i.e >3000) and use a bigger model architecture(i.e [6, 6, 6, 6])
