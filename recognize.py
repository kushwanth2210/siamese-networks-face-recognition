import torch
from torch import nn
from model import Net
from PIL import Image
import torchvision.transforms as T

# creating the neural net and loading the trained parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
architecture = [1, 1, 1, 1]
net = Net(architecture).to(device)
net.load_state_dict(torch.load('trained_model.pth.tar', map_location=device))
net.eval()

@torch.no_grad()
def recognize(img1_path, img2_path):
    '''
    takes in 2 image paths as inputs and outputs if these images are a match or not
    also outputs the distance b/w the image encodings
    '''
    transform = T.Compose(
                        [
                            T.Resize((224, 224)),
                            T.ToTensor()
                        ]
                    )
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    img1 = transform(img1).unsqueeze(0).to(device)
    img2 = transform(img2).unsqueeze(0).to(device)
    enc1, enc2 = net(img1), net(img2)
    min_dist = 1.0
    dist = torch.linalg.norm(enc1 - enc2)
    if dist < min_dist:
        print('it\'s a match!!')
        print(f'distance: {dist}')
    else:
        print('it\'s not a match!!')
        print(f'distance: {dist}')

if __name__ == '__main__':
    recognize('../test_images/0000000.jpg', '../test_images/0001_0000297_script.jpg')