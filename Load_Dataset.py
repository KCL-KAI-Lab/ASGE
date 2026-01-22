from torchvision import datasets, transforms
import torch
import torch.utils.data as data

def load_mnist(batch_size=100):
    """Return MNIST DataLoader (train_loader, test_loader)"""

    transform1 = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    transform2 = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    full_train_dataset = datasets.MNIST(root='..', train=True, download=True, transform=transform1)
    test_dataset = datasets.MNIST(root='..', train=False, download=True, transform=transform2)

    torch.manual_seed(42)

    train_dataset, val_dataset = data.random_split(full_train_dataset, [50000, 10000])

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,persistent_workers=True)
    valid_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,persistent_workers=True)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,persistent_workers=True)

    return train_loader, test_loader, valid_loader ,1

def load_fmnist(batch_size=128):
    """Return F-MNIST DataLoader (train_loader, test_loader)"""

    transform1 = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform2 = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    full_train_dataset = datasets.FashionMNIST(root='..', train=True, download=True, transform=transform1)
    test_dataset = datasets.FashionMNIST(root='..', train=False, download=True, transform=transform2)

    torch.manual_seed(42)

    train_dataset, val_dataset = data.random_split(full_train_dataset, [50000, 10000])

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,persistent_workers=True)
    valid_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,persistent_workers=True)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,persistent_workers=True)

    return train_loader, test_loader, valid_loader, 1


def load_cifar_10(batch_size=128):
    """Return cifar-10 DataLoader (train_loader, test_loader)"""


    transform1 = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
    ])

    transform2 = transforms.Compose([
        transforms.ToTensor(),
    ])

    full_train_dataset = datasets.CIFAR10(root='..', train=True, download=True, transform=transform1)
    test_dataset = datasets.CIFAR10(root='..', train=False, download=True, transform=transform2)

    torch.manual_seed(42)

    train_dataset, val_dataset = data.random_split(full_train_dataset, [45000, 5000])

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,persistent_workers=True)
    valid_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,persistent_workers=True)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,persistent_workers=True)

    return train_loader, test_loader, valid_loader, 3

def load_cifar_100(batch_size=128):
    """Return CIFAR-100 DataLoader (train_loader, test_loader)"""

    transform1 = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
    ])

    transform2 = transforms.Compose([
        transforms.ToTensor(),
    ])

    full_train_dataset = datasets.CIFAR100(root='..', train=True, download=True, transform=transform1)
    test_dataset = datasets.CIFAR100(root='..', train=False, download=True, transform=transform2)

    torch.manual_seed(42)

    train_dataset, val_dataset = data.random_split(full_train_dataset, [45000, 5000])

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,persistent_workers=True)
    valid_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,persistent_workers=True)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,persistent_workers=True)

    return train_loader, test_loader, valid_loader, 3
