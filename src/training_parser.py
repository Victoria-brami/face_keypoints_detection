import argparse


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', default='../data')
    parser.add_argument('--lr', default=1e-4)
    parser.add_argument('--modelname', default='resnet18', choices=['resnet18', 'resnet50'])
    parser.add_argument('--wd', default=1e-5, help='Weight Decay')
    parser.add_argument('--batch_size', default=16)
    parser.add_argument('--aug', nargs='+', default=["flip"])
    parser.add_argument('--n_epochs', default=30)
    parser.add_argument('--checkpoint_dir', default='../checkpoints')
    parser.add_argument('--init_from_checkpoint', default=' ')
    parser.add_argument('--train_device', default='cuda:0')
    parser.add_argument('--test_device', default='cpu')
    parser.add_argument('--num_workers', default=2)
    parser.add_argument('--optimizer', default='adam')
    parser.add_argument('--writer_folder', default='../checkpoints')
    parser.add_argument('--snapshot', default=10, help="Frequency form saving a checkpoint")

    args = parser.parse_args()
    parameters = {key: val for key, val in vars(args).items() if val is not None}

    return parameters
