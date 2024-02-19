import argparse
import sys

def set_params():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--cuda', type=bool, default=False)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0001)

    parser.add_argument('--save_path', type=str, default='dataset/data')
    parser.add_argument('--l2_coef', type=float, default=5e-4)
    parser.add_argument('--compress_ratio', type=float, default=1)
    parser.add_argument('--hidden_dim', nargs='+', type=int, default=[256, 512])
    parser.add_argument('--rel_on', type=str, default='aov')
    
    parser.add_argument('--cluster_w', type=float, default=0.5)
    parser.add_argument('--prob_v', type=float, default=0.9)
    parser.add_argument('--coa_th', type=int, default=0)
    parser.add_argument('--coo_th', type=int, default=0.5)
    parser.add_argument('--cov_th', type=float, default=2)

    parser.add_argument('--db_eps', type=float, default=0.1)
    parser.add_argument('--db_min', type=int, default=5)
    parser.add_argument('--post_match', type=bool, default=True)

    parser.add_argument('--th_a', type=list, default=[0,1])
    parser.add_argument('--th_o', type=list, default=[0.6,0.5])
    parser.add_argument('--th_v', type=list, default=[1,2])
    parser.add_argument('--repeat_num', type=int, default=1)
    
    args, _ = parser.parse_known_args()
    
    return args
