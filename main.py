import sys
import time
import argparse
import pickle
import os
from utils import *
import pandas as pd
from sessionG_diff import *

dataset = 'Sports'

def init_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000 // 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default=dataset, help='Food/Sports/Cell/Clothes')
parser.add_argument('--model', default='HIDE', help='[GCEGNN, SRGNN, DHCN, SAHNN, COTREC]')
parser.add_argument('--hiddenSize', type=int, default=200)  # best is 200
parser.add_argument('--epoch', type=int, default=25)
parser.add_argument('--activate', type=str, default='relu')
parser.add_argument('--w', type=int, default=12, help='max window size')
parser.add_argument('--gpu_id', type=str,default="0")
parser.add_argument('--batch_size', type=int, default=64)  # best is 64
parser.add_argument('--lr', type=float, default=0.002, help='learning rate.')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay.')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay.')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty ')
parser.add_argument('--layer', type=int, default=1, help='the number of layer used') #1
parser.add_argument('--n_iter', type=int, default=1)    
parser.add_argument('--seed', type=int, default=2021)
parser.add_argument('--sw_edge', default=True, help='slide_window_edge')
parser.add_argument('--item_edge', default=True, help='item_edge')
parser.add_argument('--validation', default=False, help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=6)

# multimodal
parser.add_argument('--img_features_path',default=f"Features/{dataset}_img_features.csv")
parser.add_argument('--txt_features_path', default=f"Features/{dataset}_txt_features.csv" )
parser.add_argument('--device', type=str, default='cuda', help='Device to run the diffusion model on. Choose between "cuda" and "cpu".')
parser.add_argument('--mean_type', type=str, default='x0', help='MeanType for diffusion: x0, eps')
parser.add_argument('--steps', type=int, default=5, help='diffusion steps')
parser.add_argument('--noise_schedule', type=str, default='linear-var', help='the schedule for noise generating')
parser.add_argument('--noise_scale', type=float, default=0.1, help='noise scale for noise generating') 
parser.add_argument('--noise_min', type=float, default=0.0001, help='noise lower bound for noise generating') 
parser.add_argument('--noise_max', type=float, default=0.02, help='noise upper bound for noise generating')
parser.add_argument('--sampling_noise', type=bool, default=False, help='sampling with noise or not')
parser.add_argument('--sampling_steps', type=int, default=0, help='steps of the forward process during inference')
parser.add_argument('--reweight', type=bool, default=True, help='assign different weight to different timestep or not')
parser.add_argument('--time_type', type=str, default='cat', help='cat or add')
parser.add_argument('--dims', type=list, default= [1000], help='the dims for the DNN')
parser.add_argument('--norm', type=bool, default=False, help='Normalize the input or not')
parser.add_argument('--emb_size', type=int, default=20, help='timestep embedding size')
parser.add_argument('--structure', type=str, default="MLP", help="transformer/ MLP /ResNet/ UNet" )
parser.add_argument('--num_heads', type=int, default=4, help="the number of attentation head")

# hyperparameters
parser.add_argument('--beta1', type=int, default=0.1, help='Weight for alignment ID Loss')
parser.add_argument('--alpha1', type=int, default=1, help='Weight for alignment VT loss')

opt = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id



def main():
    exp_seed = opt.seed
    top_K = [10, 20]
    init_seed(exp_seed)

    sw = []
    for i in range(2, opt.w+1):
        sw.append(i)

    
    if opt.dataset == 'Tmall':
        num_node = 40727
        opt.n_iter = 1
        opt.dropout_gcn = 0.6
        opt.dropout_local = 0.0
        opt.e = 0.4
        opt.w = 6
        sw = []
        for i in range(2, opt.w+1):
            sw.append(i)

    elif opt.dataset == 'lastfm':
        num_node = 35231
        opt.n_iter = 1
        opt.dropout_gcn = 0.1
        opt.dropout_local = 0.0

    elif opt.dataset == 'Food':
        num_node = 11638
        opt.n_iter = 1

    elif opt.dataset == 'Sports':
        num_node = 18796
        opt.n_iter = 1
    
    elif opt.dataset == 'Cell':
        num_node = 8614
        opt.n_iter = 1   
        
    elif opt.dataset == 'Clothes':
        num_node = 28196
        opt.n_iter = 1

    print(">>SEED:{}".format(exp_seed))
    # ==============================
    print('===========config================')
    print("model:{}".format(opt.model))
    print("dataset:{}".format(opt.dataset))
    print("gpu:{}".format(opt.gpu_id))
    print("item_edge:{}".format(opt.item_edge))
    print("sw_edge:{}".format(opt.sw_edge))
    print("Test Topks{}:".format(top_K))
    print(f"Slide Window:{sw}")
    print('===========end===================')
   
    datapath = r'./datasets/'
    all_train = pickle.load(open(datapath + opt.dataset + '/new_train.txt', 'rb'))
    train_data = pickle.load(open(datapath + opt.dataset + '/new_train.txt', 'rb'))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open(datapath + opt.dataset + '/new_test.txt', 'rb'))


    train_data = Data(train_data, all_train, opt, n_node=num_node, sw=sw)
    test_data = Data(test_data, all_train, opt, n_node=num_node, sw=sw)

    if opt.model == 'HIDE':
        model = trans_to_cuda(HIDE(opt, num_node))
    start = time.time()

    best_results = {}
    for K in top_K:
        best_results['epoch%d' % K] = [0, 0]
        best_results['metric%d' % K] = [0, 0]

    bad_counter = 0

    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print(f'EPOCH:{epoch}')
        print(f'Time:{time.strftime("%Y/%m/%d %H:%M:%S")}')
        metrics = train_test(model, train_data, test_data, top_K, opt)
        
        print(f"  Time Usage:")
        print(f"  Training: {metrics['train_time']:.2f}s")
        print(f"  Testing: {metrics['test_time']:.2f}s")
        print(f"  Total: {metrics['train_time'] + metrics['test_time']:.2f}s\n")
        
        start = time.time()
        for K in top_K:
            metrics['hit%d' % K] = np.mean(metrics['hit%d' % K]) * 100
            metrics['mrr%d' % K] = np.mean(metrics['mrr%d' % K]) * 100
            if best_results['metric%d' % K][0] < metrics['hit%d' % K]:
                best_results['metric%d' % K][0] = metrics['hit%d' % K]
                best_results['epoch%d' % K][0] = epoch
                flag = 1
            if best_results['metric%d' % K][1] < metrics['mrr%d' % K]:
                best_results['metric%d' % K][1] = metrics['mrr%d' % K]
                best_results['epoch%d' % K][1] = epoch
                flag = 1
        for K in top_K:
            print('Best Result:')
            print('\tP@%d: %.4f\tMRR%d: %.4f\tEpoch: %d,  %d' %
                (K, best_results['metric%d' % K][0], K, best_results['metric%d' % K][1],
                best_results['epoch%d' % K][0], best_results['epoch%d' % K][1]))
            bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break

        end = time.time()
        total_time = round(end - start, 2)
        print("Total Run time: %.2f s" % total_time)
    print('-------------------------------------------------------')
 
if __name__ == '__main__':
    main()
