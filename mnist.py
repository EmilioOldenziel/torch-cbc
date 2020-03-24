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
from torch_cbc.activations import Swish

def swish(x):
    return x * F.sigmoid(x)

class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.conv2d = ConstrainedConv2d

        self.conv1 = self.conv2d(1, 32, 3, 1)
        self.conv2 = self.conv2d(32, 64, 3, 1)
        self.conv3 = self.conv2d(64, 64, 3, 1)
        self.conv4 = self.conv2d(64, 128, 3, 1)

    def forward(self, x):
        x = swish(self.conv2(swish(self.conv1(x))))
        x = F.max_pool2d(x, 2, 2)
        x = swish(self.conv4(swish(self.conv3(x))))
        x = F.max_pool2d(x, 2, 2)
        return x

import cv2 as cv
def v_test(epoch, model):
    for idx, c in enumerate(model.components):
        prototype = c
        img = prototype.view(28, 28).cpu().data.numpy()
        img = img * 255
        img = cv.resize(img, (56, 56))
        cv.imwrite("char_img/%d_%d.png" % (epoch, idx), img)

def train(args, model, device, train_loader, optimizer, lossfunction, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        onehot = torch.zeros(len(target), 10, device=device) \
                      .scatter_(1, target.unsqueeze(1), 1.)  # 10 classes
        loss = lossfunction(output, onehot).mean() #it seems mean better than sum
        loss.backward()
        optimizer.step()

        # Clamp reasoning and components
        for name, p in model.named_parameters():
            if 'reasoning' in name or 'components' in name:
                p.data.clamp_(0)

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

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(  # noqa
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


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
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
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
                     n_components=9,
                     component_shape=(1, 28, 28)).to(device)

    print(model)

    #backbone_params = [p for name, p in model.named_parameters() if not 'components' in name]
    #component_params = [p for name, p in model.named_parameters() if not 'backbone' in name]

    #optimizer2 = optim.Adam(backbone_params, lr=args.lr)
    #optimizer1 = optim.Adam(component_params, lr=args.lr)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    lossfunction = MarginLoss()

    print("Starting training")
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, lossfunction, epoch)  # noqa
        test(args, model, device, test_loader, lossfunction)
        v_test(epoch, model)

    if (args.save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
