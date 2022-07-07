import torch.optim
import torch

from alig import AliG
from borat import BORAT
from alig import l2_projection


def get_optimizer(args, model, loss, parameters):
    parameters = list(parameters)
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=args.eta, weight_decay=args.weight_decay,
                                    momentum=args.momentum, nesterov=bool(args.momentum))
    elif args.opt == 'alig':
        optimizer = AliG(parameters, max_lr=args.eta, momentum=args.momentum,
                         projection_fn=lambda: l2_projection(parameters, args.max_norm))
    elif args.opt == 'borat':
        optimizer = BORAT(parameters, model, loss, eta=args.eta, n=args.n, momentum=args.momentum,
                         projection_fn=lambda: l2_projection(parameters, args.max_norm), sgd_forward=args.sgdf, same_batch=args.same_batch, debug=args.debug)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(parameters, lr=args.eta, weight_decay=args.weight_decay)
    else:
        raise ValueError(args.opt)

    print("Optimizer: \t {}".format(args.opt.upper()))

    optimizer.step_size = args.eta
    optimizer.step_size_unclipped = args.eta
    optimizer.momentum = args.momentum
    optimizer.step_0 = 0
    optimizer.step_1 = 0
    optimizer.step_2 = 0
    optimizer.step_3 = 0
    optimizer.step_4 = 0

    if args.load_opt:
        state = torch.load(args.load_opt)['optimizer']
        optimizer.load_state_dict(state)
        print('Loaded optimizer from {}'.format(args.load_opt))

    return optimizer


def decay_optimizer(optimizer, decay_factor=0.1):
    if isinstance(optimizer, torch.optim.SGD):
        for param_group in optimizer.param_groups:
            param_group['lr'] *= decay_factor

        optimizer.step_size = optimizer.param_groups[0]['lr']
        optimizer.step_size_unclipped = optimizer.param_groups[0]['lr']
    else:
        raise ValueError
