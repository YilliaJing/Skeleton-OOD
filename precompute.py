import random
import sys
import argparse
import torch
import torch.nn.functional as F
import yaml
import numpy as np
from torchlight import DictAction
from tqdm import tqdm
from feeders.feeder_ntu import Feeder as FeederNTU
import os

torch.cuda.empty_cache()

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))

def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Spatial Temporal Graph Convolution Network')
    parser.add_argument(
        '--work-dir',
        default='/work_dir/ntu_hdgcn/cross-label/joint_CoM_1',
        help='the work folder for storing results')

    parser.add_argument('-model_saved_name', default='')
    parser.add_argument(
        '--config',
        default='./work_dir/ntu_hdgcn/cross-label/joint_CoM_seen-55/config-mix.yaml',
        help='path to the configuration file')  # 修改config文件

    # processor
    parser.add_argument(
        '--phase', default='test', help='must be train or test')  # 训练/测试阶段
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=True,
        help='if ture, the classification score will be stored')  # 分类分数保存

    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=1,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--save-epoch',
        type=int,
        default=30,
        help='the start epoch to save model (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=32,
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        action=DictAction,
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        action=DictAction,
        default=dict(),
        help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        action=DictAction,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default='',
        help='the weights for network initialization')  # 读取训练好的模型weights  在config里面要写上
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=1,
        nargs='+',
        help='the indexes of GPUs for training or testing')  # gpu
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument(
        '--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument(
        '--lr-ratio',
        type=float,
        default=0.001,
        help='decay rate for learning rate')
    parser.add_argument(
        '--lr-decay-rate',
        type=float,
        default=0.1,
        help='decay rate for learning rate')
    parser.add_argument('--warm_up_epoch', type=int, default=0)
    parser.add_argument('--loss_type', type=str, default='CE')

    return parser

if __name__ == '__main__':
    print(os.getcwd())
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)
    arg = parser.parse_args()
    print(arg.config)

    Model = import_class('model.HDGCN.Model')
    net = Model(**arg.model_args)

    data_loader = dict()
    data_loader['train'] = torch.utils.data.DataLoader(
        dataset=FeederNTU(**arg.train_feeder_args),
        batch_size=arg.batch_size,
        shuffle=True,
        num_workers=arg.num_worker,
        drop_last=True,
        worker_init_fn=init_seed)
    data_loader['test'] = torch.utils.data.DataLoader(
        dataset=FeederNTU(**arg.test_feeder_args),
        batch_size=arg.test_batch_size,
        shuffle=False,
        num_workers=arg.num_worker,
        drop_last=False,
        worker_init_fn=init_seed)

    # output_device = 3
    # net = net.cuda(output_device)
    net.eval()
    process = tqdm(data_loader['train'])

    feat_log = np.zeros((46898, 256))  # size_train_data, feature_dim
    '''score_log = np.zeros((46898, arg.model_args['num_class']))
    label_log = np.zeros(46898)'''
    for batch_idx, (data, label, index) in enumerate(process):
        inputs = data.float()  # .cuda(output_device)
        targets = label.long()  # .cuda(output_device)
        # inputs, targets = data.to(output_device), label.to(output_device)  # 64, 3, 64, 25, 2
        start_ind = batch_idx * arg.batch_size
        end_ind = min((batch_idx + 1) * arg.batch_size, len(data_loader['train'].dataset))

        _, feature = net(inputs)  # fc换成dice的？内存问题？
        feat_log[start_ind:end_ind, :] = feature.data.cpu().numpy()
        if batch_idx % 10 == 0:
            print(f"{batch_idx}/{len(process)}")

    np.save(f"hdgcn_55_feat_stat.npy", feat_log.mean(0))
    print("done")