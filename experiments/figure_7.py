from functools import lru_cache
import train_classifier
from generate_data import *
from dataloader import *
import json

cf_ratios = [1, 5, 10, 20]
dataset_sizes = [10**4, 10**5, 10**6]
datasets = ['colored_MNIST', 'double_colored_MNIST', 'wildlife_MNIST']
file = "final_results.json"

def generate_dataset(dataset, cf_ratio, size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cgn = CGN()
    cgn.load_state_dict(torch.load("mnists/experiments/cgn_" + dataset + "/weights/ckp.pth", 'cpu'))
    cgn.to(device).eval()

    print(f"Generating the counterfactual {dataset} of size {size} with cf_ratio {cf_ratio}")
    generate_cf_dataset(cgn=cgn, path=dataset + '_counterfactual.pth',
                        dataset_size=size, no_cfs=cf_ratio,
                        device=device, counterfactual=True)

def train_(dataset, args):
    args.dataset = dataset
    result = train_classifier.main(args)
    return result

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=TENSOR_DATASETS,
                    help='Provide dataset name.')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 14)')
parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                    help='learning rate (default: 1.0)')
parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                    help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--original', default=False, action='store_true',
                    help='original data (True) or no original data (False)')
args = parser.parse_args()
args.original = True

results = {}
for dataset in datasets:
    sizes = {}
    for size in dataset_sizes:
        ratios = {}
        for cf_ratio in cf_ratios:
            ratios[cf_ratio] = None
            sizes[size] = ratios
            results[dataset] = sizes

for dataset in datasets:
    for size in dataset_sizes:
        for cf_ratio in cf_ratios:
            if not (dataset == "colored_MNIST" and cf_ratio == 20):
                generate_dataset(dataset, cf_ratio, size)
                result = train_(dataset + "_counterfactual", args)
                results[dataset][size][cf_ratio] = result

                with open('json_data.json', 'w') as f:
                    json.dump(results, f)



