import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # Federated Learning related arguments
    parser.add_argument('--epochs', type=int, default=30, help="Number of rounds of training")
    parser.add_argument('--num_users', type=int, default=10, help="Number of clients")
    parser.add_argument('--frac', type=float, default=1.0, help='The fraction of clients to use in update (default 1)')
    parser.add_argument('--local_ep', type=int, default=10, help="The number of local epochs")
    parser.add_argument('--local_bs', type=int, default=50, help="Local batch size")
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default 0.001)')  # 0.001
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default 0.9)')  # 0.9

    # other arguments
    parser.add_argument('--gpu', default=0, help="GPU ID")
    parser.add_argument('--optimizer', type=str, default='sgd', help="Type of optimizer (sgd, adam)")
    parser.add_argument('--iid', type=int, default=1, help='1 for IID, 0 for non-IID (default 1)')
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # custom arguments
    parser.add_argument('--noise', type=str, default='gaussian',
                        help='Give noise (gaussian, saltpepper, speckle')
    parser.add_argument('--noise_frac', type=float, default=0.2,
                        help='Give noise in data to [X] proportion of users')
    parser.add_argument('--clean', type=int, default=2, help='Number of noised users to find')
    parser.add_argument('--file', type=str, help="Option JSON file name")

    args = parser.parse_args()
    return args
