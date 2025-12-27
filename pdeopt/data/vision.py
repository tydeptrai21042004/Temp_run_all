import torch

def get_mnist_loaders(batch_size=128, num_workers=2):
    import torchvision
    import torchvision.transforms as T
    tfm = T.Compose([T.ToTensor()])
    train_ds = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test_ds = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=tfm)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader

def get_fashionmnist_loaders(batch_size=128, num_workers=2):
    import torchvision
    import torchvision.transforms as T
    tfm = T.Compose([T.ToTensor()])
    train_ds = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=tfm)
    test_ds = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=tfm)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader

def get_cifar10_loaders(batch_size=128, num_workers=2):
    import torchvision
    import torchvision.transforms as T
    tfm = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train_ds = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=tfm)
    test_ds = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=tfm)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader
