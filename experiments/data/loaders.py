import os

import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from .balancedsampler import BalancedBatchSampler

from .utils import random_subsets, Subset

def create_loaders(dataset_train, dataset_val, dataset_test,
                   train_size, val_size, test_size, batch_size, test_batch_size,
                   cuda, n_classes, num_workers, split=True, eq_class=False):

    kwargs = {'num_workers': num_workers, 'pin_memory': True} if cuda else {}

    if split:
        train_indices, val_indices = random_subsets((train_size, val_size),
                                                    len(dataset_train),
                                                    seed=1234)
    else:
        train_size = train_size if train_size is not None else len(dataset_train)
        train_indices, = random_subsets((train_size,),
                                        len(dataset_train),
                                        seed=1234)
        val_size = val_size if val_size is not None else len(dataset_val)
        val_indices, = random_subsets((val_size,),
                                      len(dataset_val),
                                      seed=1234)

    test_size = test_size if test_size is not None else len(dataset_test)
    test_indices, = random_subsets((test_size,),
                                   len(dataset_test),
                                   seed=1234)

    dataset_train = Subset(dataset_train, train_indices)
    dataset_val = Subset(dataset_val, val_indices)
    dataset_test = Subset(dataset_test, test_indices)

    print('Dataset sizes: \t train: {} \t val: {} \t test: {}'
          .format(len(dataset_train), len(dataset_val), len(dataset_test)))
    print('Batch size: \t {}'.format(batch_size))

    if eq_class:
        assert batch_size % n_classes == 0
        balanced_batch_sampler = BalancedBatchSampler(dataset_train,
                                                      n_classes, batch_size//n_classes)
        train_loader = data.DataLoader(dataset_train,
                                       batch_sampler=balanced_batch_sampler,
                                       **kwargs)
    else:
        train_loader = data.DataLoader(dataset_train,
                                       batch_size=batch_size,
                                       shuffle=True, **kwargs)

    val_loader = data.DataLoader(dataset_val,
                                 batch_size=test_batch_size,
                                 shuffle=False, **kwargs)

    test_loader = data.DataLoader(dataset_test,
                                  batch_size=test_batch_size,
                                  shuffle=False, **kwargs)

    train_loader.tag = 'train'
    val_loader.tag = 'val'
    test_loader.tag = 'test'

    return train_loader, val_loader, test_loader

def loaders_mnist(dataset, batch_size=64, cuda=0,
                  train_size=50000, val_size=10000, test_size=10000,
                  test_batch_size=1000, augment=False, **kwargs):

    assert dataset == 'mnist'
    root = '{}/{}'.format(os.environ['VISION_DATA'], dataset)

    # Data loading code
    normalize = transforms.Normalize(mean=(0.1307,),
                                     std=(0.3081,))

    transform = transforms.Compose([transforms.ToTensor(), normalize])

    # define two datasets in order to have different transforms
    # on training and validation
    dataset_train = datasets.MNIST(root=root, train=True, transform=transform)
    dataset_val = datasets.MNIST(root=root, train=True, transform=transform)
    dataset_test = datasets.MNIST(root=root, train=False, transform=transform)

    return create_loaders(dataset_train, dataset_val,
                          dataset_test, train_size, val_size, test_size,
                          batch_size=batch_size,
                          test_batch_size=test_batch_size,
                          cuda=cuda, num_workers=0)

def loaders_cifar(dataset, batch_size, cuda, n_classes, eq_class,
                  train_size=45000, augment=True, val_size=5000, test_size=10000,
                  test_batch_size=128, **kwargs):

    assert dataset in ('cifar10', 'cifar100')

    root = '{}/{}'.format(os.environ['VISION_DATA'], dataset)

    # Data loading code
    mean = [125.3, 123.0, 113.9]
    std = [63.0, 62.1, 66.7]
    normalize = transforms.Normalize(mean=[x / 255.0 for x in mean],
                                     std=[x / 255.0 for x in std])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize])

    if augment:
        print('Using data augmentation')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
    else:
        print('Not using data augmentation')
        transform_train = transform_test

    # define two datasets in order to have different transforms
    # on training and validation (no augmentation on validation)
    dataset = datasets.CIFAR10 if dataset == 'cifar10' else datasets.CIFAR100
    dataset_train = dataset(root=root, train=True, download=True,
                            transform=transform_train)
    dataset_val = dataset(root=root, train=True,
                          transform=transform_test)
    dataset_test = dataset(root=root, train=False, download=True,
                           transform=transform_test)

    return create_loaders(dataset_train, dataset_val,
                          dataset_test, train_size, val_size, test_size,
                          batch_size, test_batch_size, cuda, n_classes, num_workers=2, eq_class=eq_class)

def loaders_svhn(dataset, batch_size, cuda,
                 train_size=None, augment=False, val_size=6000, test_size=26032,
                 test_batch_size=1000, **kwargs):

    assert 'svhn' in dataset

    root = '{}/svhn'.format(os.environ['VISION_DATA'])
    dataset_name = dataset

    # Data loading code
    transform_test = transforms.Compose([
        transforms.ToTensor()])

    if augment:
        assert ValueError("Should not be using data augmentation on SVHN")
    else:
        print('Not using data augmentation')
        transform_train = transform_test

    # define two datasets in order to have different transforms
    # on training and validation (no augmentation on validation)
    dataset = datasets.SVHN
    if dataset_name == 'svhn':
        train_size = 73257 - val_size if train_size is None else train_size
        dataset_train = dataset(root=root, split='train', download=True,
                                transform=transform_train)
        dataset_val = dataset(root=root, split='train',
                              transform=transform_test)
        split = True
    elif dataset_name == 'svhn-extra':
        # manual train-val split within difficult examples (i.e. not from extra data)
        train_indices, val_indices = random_subsets((73257 - val_size, val_size), 73257, seed=1234)

        # not-extra data: split betwwen train and val
        dataset_train_reduced = Subset(dataset(root=root, split='train', transform=transform_train, download=True), train_indices)
        dataset_val = Subset(dataset(root=root, split='train', transform=transform_test), val_indices)

        # add extra data to train
        dataset_train = data.ConcatDataset((dataset_train_reduced, dataset(root=root, split='extra', transform=transform_train, download=True)))
        split = False
    else:
        raise ValueError
    dataset_test = dataset(root=root, split='test', download=True,
                           transform=transform_test)

    return create_loaders(dataset_train, dataset_val,
                          dataset_test, train_size, val_size, test_size,
                          batch_size, test_batch_size, cuda, n_classes, num_workers=4, split=split)

def loaders_imagenet(dataset, batch_size, cuda,
                                          train_size=1231166, augment=True, val_size=50000,
                                          test_size=50000, test_batch_size=256, topk=None, noise=False,
                                          multiple_crops=False, data_root=None, **kwargs):
    assert dataset == 'imagenet'
    data_root = data_root if data_root is not None else os.environ['VISION_DATA_SSD']
    root = '{}/ILSVRC2012-prepr-split/images'.format(data_root)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    traindir = os.path.join(root, 'train')
    valdir = os.path.join(root, 'val')
    testdir = os.path.join(root, 'test')
    normalize = transforms.Normalize(mean=mean, std=std)
    if multiple_crops:
        print('Using multiple crops')
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.TenCrop(224),
            lambda x: [normalize(transforms.functional.to_tensor(img)) for img in x]])
    else:
        transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])
    if augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
    else:
        transform_train = transform_test
    dataset_train = datasets.ImageFolder(traindir, transform_train)
    dataset_val = datasets.ImageFolder(valdir, transform_test)
    dataset_test = datasets.ImageFolder(testdir, transform_test)
    return create_loaders(dataset_train, dataset_val,
                          dataset_test, train_size, val_size, test_size,
                          batch_size, test_batch_size, cuda, n_classes, num_workers=8, split=False)

