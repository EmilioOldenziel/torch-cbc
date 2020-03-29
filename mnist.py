from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from torch_cbc.cbc_model import CBCModel
from torch_cbc.losses import MarginLoss
from torch_cbc.constraints import EuclideanNormalization
from torch_cbc.layers import ConstrainedConv2d
from torch_cbc.activations import swish

from utils import visualize_components

import numpy as np


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.conv2d = ConstrainedConv2d

        self.conv1 = self.conv2d(1, 32, 3, 1)
        self.conv2 = self.conv2d(32, 64, 3, 1)
        self.conv3 = self.conv2d(64, 64, 3, 1)
        self.conv4 = self.conv2d(64, 128, 3, 1)
        self.maxpool2d = nn.MaxPool2d(2)

    def forward(self, x):
        x = swish(self.conv2(swish(self.conv1(x))))
        x = self.maxpool2d(x)
        x = swish(self.conv4(swish(self.conv3(x))))
        x = self.maxpool2d(x)
        return x


def train(args, model, device, train_loader, optimizer, lossfunction, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        onehot = torch.zeros(len(target), 10, device=device) \
                      .scatter_(1, target.unsqueeze(1), 1.)  # 10 classes
        loss = lossfunction(output, onehot).mean()
        loss.backward()
        optimizer.step()

        for name, p in model.named_parameters():
            if ('components' in name) or ('reasoning' in name):
                p.data.clamp_(0, 1)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader, lossfunction):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            onehot = torch.zeros(len(target), 10, device=device) \
                          .scatter_(1, target.unsqueeze(1), 1.)  # 10 classes
            test_loss += lossfunction(output, onehot).sum().item()  # sum up batch loss # noqa
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability # noqa
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(  # noqa
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return test_loss


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',  # noqa
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.003, metavar='LR',
                        help='learning rate (default: 0.003)')
    parser.add_argument('--margin', type=float, default=0.3,
                        help='Margin Loss margin (default: 0.3)')
    parser.add_argument('--n_components', type=int, default=9, metavar='N',
                        help='number of components (default: 9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')  # noqa
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.RandomAffine(0, translate=(1/14, 1/14)),
                           transforms.RandomRotation(15, fill=(0,)),
                           transforms.ToTensor()
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    backbone = Backbone()
    model = CBCModel(backbone,
                     n_classes=10,
                     n_components=args.n_components,
                     component_shape=(1, 28, 28))
    
    model.backbone.conv1.weight.data = torch.as_tensor(np.load("./keras_weights/conv2d_1_weights.npy"), dtype=torch.float32).permute(3,2,1,0)
    model.backbone.conv1.bias.data = torch.as_tensor(np.load("./keras_weights/conv2d_1_bias.npy"), dtype=torch.float32)

    model.backbone.conv2.weight.data = torch.as_tensor(np.load("./keras_weights/conv2d_2_weights.npy"), dtype=torch.float32).permute(3,2,1,0)
    model.backbone.conv2.bias.data = torch.as_tensor(np.load("./keras_weights/conv2d_2_bias.npy"), dtype=torch.float32)

    model.backbone.conv3.weight.data = torch.as_tensor(np.load("./keras_weights/conv2d_3_weights.npy"), dtype=torch.float32).permute(3,2,1,0)
    model.backbone.conv3.bias.data = torch.as_tensor(np.load("./keras_weights/conv2d_3_bias.npy"), dtype=torch.float32)

    model.backbone.conv4.weight.data = torch.as_tensor(np.load("./keras_weights/conv2d_4_weights.npy"), dtype=torch.float32).permute(3,2,1,0)
    model.backbone.conv4.bias.data = torch.as_tensor(np.load("./keras_weights/conv2d_4_bias.npy"), dtype=torch.float32)

    #print(torch.as_tensor(np.load("./keras_weights/add_components_1_weights.npy"), dtype=torch.float32).squeeze(0).permute(0,3,1,2).shape)
    #print(model.components.shape)
    model.components.data = torch.as_tensor(np.load("./keras_weights/add_components_1_weights.npy"), dtype=torch.float32).squeeze(0).permute(0,3,1,2)

    for name, p in model.named_parameters():
        if ("backbone" in name) or ("components" in name):
            p.requires_grad = False
        else:
            print(name)

    model.to(device)

    

    print(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 
                                                     patience=5, factor=0.9)

    lossfunction = MarginLoss(margin=args.margin)

    print("Starting training")
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, lossfunction, epoch)  # noqa
        test_loss = test(args, model, device, test_loader, lossfunction)
        scheduler.step(test_loss)
        visualize_components(epoch, model, "./visualzation")

    if (args.save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
