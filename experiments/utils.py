import os
import sys
import socket
import torch
import mlogger
import random
import numpy as np

def regularization(model, l2):
    reg = 0.5 * l2 * sum([p.data.norm() ** 2 for p in model.parameters()]) if l2 else 0
    return reg


def set_seed(args, print_out=True):
    if args.seed is None:
        np.random.seed(None)
        args.seed = np.random.randint(1e5)
    if print_out:
        print('Seed:\t {}'.format(args.seed))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)


def setup_xp(args, model, optimizer):

    env_name = args.xp_name.split('/')[-1]
    if args.visdom:
        plotter = mlogger.VisdomPlotter({'env': env_name, 'server': args.server, 'port': args.port})
    else:
        plotter = None

    xp = mlogger.Container()

    xp.config = mlogger.Config(plotter=plotter, **vars(args))

    xp.epoch = mlogger.metric.Simple()

    xp.train = mlogger.Container()
    xp.train.acc = mlogger.metric.Average(plotter=plotter, plot_title="Accuracy", plot_legend="training")
    xp.train.loss = mlogger.metric.Average(plotter=plotter, plot_title="Objective", plot_legend="loss")
    xp.train.obj = mlogger.metric.Simple(plotter=plotter, plot_title="Objective", plot_legend="objective")
    xp.train.reg = mlogger.metric.Simple(plotter=plotter, plot_title="Objective", plot_legend="regularization")
    xp.train.weight_norm = mlogger.metric.Simple(plotter=plotter, plot_title="Weight-Norm")
    xp.train.step_size = mlogger.metric.Average(plotter=plotter, plot_title="Step-Size", plot_legend="clipped")
    xp.train.step_size_u = mlogger.metric.Average(plotter=plotter, plot_title="Step-Size", plot_legend="unclipped")
    xp.train.timer = mlogger.metric.Timer(plotter=plotter, plot_title="Time", plot_legend='training')

    xp.val = mlogger.Container()
    xp.val.acc = mlogger.metric.Average(plotter=plotter, plot_title="Accuracy", plot_legend="validation")
    xp.val.timer = mlogger.metric.Timer(plotter=plotter, plot_title="Time", plot_legend='validation')

    xp.max_val = mlogger.metric.Maximum(plotter=plotter, plot_title="Accuracy", plot_legend='best-validation')

    xp.test = mlogger.Container()
    xp.test.acc = mlogger.metric.Average(plotter=plotter, plot_title="Accuracy", plot_legend="test")
    xp.test.timer = mlogger.metric.Timer(plotter=plotter, plot_title="Time", plot_legend='test')

    xp.train.alpha0 = mlogger.metric.Average(plotter=plotter, plot_title="Step-Type", plot_legend="alpha0")
    xp.train.alpha1 = mlogger.metric.Average(plotter=plotter, plot_title="Step-Type", plot_legend="alpha1")
    xp.train.alpha2 = mlogger.metric.Average(plotter=plotter, plot_title="Step-Type", plot_legend="alpha2")
    xp.train.alpha3 = mlogger.metric.Average(plotter=plotter, plot_title="Step-Type", plot_legend="alpha3")
    xp.train.alpha4 = mlogger.metric.Average(plotter=plotter, plot_title="Step-Type", plot_legend="alpha4")

    if args.dataset == "imagenet":
        xp.train.acc5 = mlogger.metric.Average(plotter=plotter, plot_title="Accuracy@5", plot_legend="training")
        xp.val.acc5 = mlogger.metric.Average(plotter=plotter, plot_title="Accuracy@5", plot_legend="validation")
        xp.test.acc5 = mlogger.metric.Average(plotter=plotter, plot_title="Accuracy@5", plot_legend="test")

    if args.visdom:
        plotter.set_win_opts("Step-Size", {'ytype': 'log'})
        plotter.set_win_opts("Objective", {'ytype': 'log'})

    if args.log:
        # log at each epoch
        xp.epoch.hook_on_update(lambda: xp.save_to('{}/results.json'.format(args.xp_name)))
        xp.epoch.hook_on_update(lambda: save_state(model, optimizer, '{}/model.pkl'.format(args.xp_name)))

        # log after final evaluation on test set
        xp.test.acc.hook_on_update(lambda: xp.save_to('{}/results.json'.format(args.xp_name)))
        xp.test.acc.hook_on_update(lambda: save_state(model, optimizer, '{}/model.pkl'.format(args.xp_name)))

        # save results and model for best validation performance
        # xp.max_val.hook_on_new_max(lambda: save_state(model, optimizer, '{}/best_model.pkl'.format(args.xp_name)))
        xp.max_val.hook_on_new_max(lambda: save_state(model, optimizer, '{}/best_model.pkl'.format(args.xp_name)))

    return xp


def save_state(model, optimizer, filename):
    torch.save({'model': model.state_dict(),
                'optimizer': optimizer.state_dict()}, filename)


def write_results(args, xp):
    with open('results.txt', 'a') as results:
        results.write('{xp_name},Train Acc,{tracc:.4f},Val Acc,{vacc:.4f},Test Acc,{teacc:.4f}\n'
                .format(xp_name=args.xp_name,
                        tracc=xp.train.acc.value,
                        vacc=xp.max_val.value,
                        teacc=xp.test.acc.value))

@torch.autograd.no_grad()
def accuracy(out, targets, topk=1):
    if topk == 1:
        _, pred = torch.max(out, 1)
        acc = torch.mean(torch.eq(pred, targets).float())
    else:
        _, pred = out.topk(topk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        acc = correct[:topk].view(-1).float().sum(0) / out.size(0)

    return 100. * acc

