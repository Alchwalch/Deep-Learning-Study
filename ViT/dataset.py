import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

__all__=[
    'prepare_cifar10',
]

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)


def get_transforms(mean=CIFAR10_MEAN, std=CIFAR10_STD, img_size=32, crop_padding=4):
    train_tf = transforms.Compose([
        transforms.RandomCrop(img_size, padding=crop_padding),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, eval_tf


def get_cifar10_datasets(data_root, train_tf, eval_tf, n_val=5000, split_seed=67):
    # train/val은 같은 원본이지만 transform이 달라야 해서 데이터셋을 두 개 만듦
    full_train = torchvision.datasets.CIFAR10(data_root, train=True,  download=True, transform=train_tf)
    full_val   = torchvision.datasets.CIFAR10(data_root, train=True,  download=True, transform=eval_tf)
    test_set   = torchvision.datasets.CIFAR10(data_root, train=False, download=True, transform=eval_tf)

    # 같은 인덱스 분할을 두 데이터셋에 적용 (val에는 augmentation이 안 들어가도록)
    perm = torch.randperm(len(full_train),
                          generator=torch.Generator().manual_seed(split_seed)).tolist()
    train_set = Subset(full_train, perm[n_val:])
    val_set   = Subset(full_val,   perm[:n_val])

    return train_set, val_set, test_set, full_train.classes


def get_dataloaders(train_set, val_set, test_set, batch_size=128, num_workers=2, base_seed=67):
    # train_model이 train_loader.generator.manual_seed(...)를 쓰므로 generator 지정 필수
    g = torch.Generator()
    g.manual_seed(base_seed)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=g,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


def prepare_cifar10(data_root='/content/data', batch_size=128, n_val=5000,
                    num_workers=2, base_seed=67):
    # 한 번에 다 만들어주는 래퍼
    train_tf, eval_tf = get_transforms()
    train_set, val_set, test_set, class_names = get_cifar10_datasets(
        data_root, train_tf, eval_tf, n_val=n_val, split_seed=base_seed)
    train_loader, val_loader, test_loader = get_dataloaders(
        train_set, val_set, test_set, batch_size, num_workers, base_seed)
    return train_loader, val_loader, test_loader, class_names