import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from torch_cbc.cbc_model import CBCModel
from torch_cbc.losses import MarginLoss
from torch_cbc.layers import ConstrainedConv2d

from utils import visualize_components


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.conv2d = ConstrainedConv2d
        self.activation = nn.Hardswish()

        self.conv1 = self.conv2d(1, 32, 3, 1)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.zeros_(self.conv1.bias)
        self.conv2 = self.conv2d(32, 64, 3, 1)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.zeros_(self.conv2.bias)
        self.conv3 = self.conv2d(64, 64, 3, 1)
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        torch.nn.init.zeros_(self.conv3.bias)
        self.conv4 = self.conv2d(64, 128, 3, 1)
        torch.nn.init.xavier_uniform_(self.conv4.weight)
        torch.nn.init.zeros_(self.conv4.bias)
        self.maxpool2d = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.activation(self.conv2(self.activation(self.conv1(x))))
        x = self.maxpool2d(x)
        x = self.activation(self.conv4(self.activation(self.conv3(x))))
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
    parser.add_argument('--n_components', type=int, default=9, metavar='C',
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
                           transforms.RandomAffine(0, 
                                                   translate=(0.1, 0.1)),
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
                     component_shape=(1, 28, 28)).to(device)

    print(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                     patience=3,
                                                     factor=0.9,
                                                     verbose=True)

    lossfunction = MarginLoss(margin=args.margin)

    print("Starting training")
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, lossfunction, epoch)  # noqa
        test_loss = test(args, model, device, test_loader, lossfunction)
        scheduler.step(test_loss)
        visualize_components(epoch, model, "./visualization")

    if (args.save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
