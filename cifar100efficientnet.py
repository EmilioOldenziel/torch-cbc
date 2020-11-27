from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet18

from torch_cbc.fixedCBC_model import FixedCBCModel
from torch_cbc.losses import MarginLoss

from utils import visualize_components

from efficientnet_pytorch import EfficientNet


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.model = EfficientNet.from_pretrained("efficientnet-b0")

    def forward(self, x):
        return self.model.extract_features(x)


def train(args, model, device, train_loader, optimizer, lossfunction, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        onehot = torch.zeros(len(target), 100, device=device) \
                      .scatter_(1, target.unsqueeze(1), 1.)  # 10 classes
        loss = lossfunction(output, onehot).mean()
        loss.backward()
        optimizer.step()

        for name, p in model.named_parameters():
            if ('reasoning' in name):
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
            onehot = torch.zeros(len(target), 100, device=device) \
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
    parser = argparse.ArgumentParser(description='PyTorch CIFAR Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=500, metavar='N',  # noqa
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--margin', type=float, default=0.3,
                        help='Margin Loss margin (default: 0.3)')
    parser.add_argument('--n_components', type=int, default=5, metavar='C',
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
        datasets.CIFAR10('../data', train=True, download=True,
                         transform=transforms.Compose([
                            transforms.RandomAffine(0,
                                                    translate=(0.1, 0.1)),
                            # transforms.RandomRotation(15, fill=(0,)),
                            transforms.Resize(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=False, transform=transforms.Compose([
                           transforms.Resize(224),
                           transforms.ToTensor(),
                           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    backbone = Backbone()

     # freeze backbone and components
    for name, p in backbone.named_parameters():
      p.requires_grad = False

    model = FixedCBCModel(backbone,
                          n_classes=100,
                          n_components=(args.n_components),
                          component_shape=(3, 224, 224),
                          data=train_loader.dataset.data,
                          targets=train_loader.dataset.targets,
                          image_transform=test_loader.dataset.transform
                          )

    visualize_components(0, model, "./visualization")

    # freeze backbone and components
    for name, p in model.named_parameters():
        if ('backbone' in name) or ('component' in name):
            p.requires_grad = False

    model.to(device)

    print(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
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
        torch.save(model.state_dict(), "cifar100_cnn.pt")


if __name__ == '__main__':
    main()
