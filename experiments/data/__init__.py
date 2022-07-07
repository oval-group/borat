from data.loaders import loaders_cifar, loaders_mnist, loaders_svhn, loaders_imagenet, loaders_tiny_imagenet

def get_data_loaders(args):

    print('Dataset: \t {}'.format(args.dataset.upper()))

    # remove values if None
    for k in ('train_size', 'val_size', 'test_size'):
        if args.__dict__[k] is None:
            args.__dict__.pop(k)

    if args.dataset == 'mnist':
        loader_train, loader_val, loader_test = loaders_mnist(**vars(args))
    elif 'cifar' in args.dataset:
        loader_train, loader_val, loader_test = loaders_cifar(**vars(args))
    elif 'svhn' in args.dataset:
        loader_train, loader_val, loader_test = loaders_svhn(**vars(args))
    elif args.dataset == 'tiny_imagenet':
        loader_train, loader_val, loader_test = loaders_tiny_imagenet(**vars(args))
    elif args.dataset == 'imagenet':
        loader_train, loader_val, loader_test = loaders_imagenet(**vars(args))
    else:
        raise NotImplementedError

    args.train_size = len(loader_train.dataset)
    args.val_size = len(loader_val.dataset)
    args.test_size = len(loader_test.dataset)

    return loader_train, loader_val, loader_test
