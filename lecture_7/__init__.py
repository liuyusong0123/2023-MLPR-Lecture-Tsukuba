import torch
import torchvision


# Load Cifar-10 data and create training/validation dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=10,
                                              shuffle=True)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True)
validation_loader = torch.utils.data.DataLoader(testset, batch_size=10,
                                             shuffle=False)

print(train_loader)
# MLP model
class MLP_Net(torch.nn.Module):
    def __init__(self):
        super(MLP_Net, self).__init__()
        self.






