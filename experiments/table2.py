import train_classifier
from generate_data import *
from dataloader import *
import json

def generate_dataset(dataset, cf_ratio, size, counterfactual):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cgn = CGN()
    cgn.load_state_dict(torch.load("mnists/experiments/cgn_" + dataset + "/weights/ckp.pth", 'cpu'))
    cgn.to(device).eval()

    print(f"Generating the counterfactual {dataset} of size {size} with cf_ratio {cf_ratio}")  
    generate_cf_dataset(cgn=cgn, path=dataset + '_counterfactual.pth',
                        dataset_size=size, no_cfs=cf_ratio,
                        device=device, counterfactual=counterfactual)
def train_(dataset, args):
    args.dataset = dataset
    result = train_classifier.main(args)
    return result

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=TENSOR_DATASETS,
                    help='Provide dataset name.')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 14)')
parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                    help='learning rate (default: 1.0)')
parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                    help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--original', default=False, action='store_true',
                    help='original data (True) or no original data (False)')
parser.add_argument('--N', type=int, default=100000)
args = parser.parse_args()

table2 = {}
json_file = "table2_data_2.json"

# Train on original + CGN
row = "original + CGN"
O_CGN = {}
args.original = True

generate_dataset("colored_MNIST", 1, args.N, True)
O_CGN["colored_MNIST"] = train_("colored_MNIST_counterfactual", args)

generate_dataset("double_colored_MNIST", 1, args.N, True)
O_CGN["double_colored_MNIST"] = train_("double_colored_MNIST_counterfactual", args)

generate_dataset("wildlife_MNIST", 1, args.N, True)
O_CGN["wildlife_MNIST"] = train_("wildlife_MNIST_counterfactual", args)

table2[row] = O_CGN

with open(json_file, 'w') as f:
    json.dump(table2, f)

# Train on CGN
row = "CGN"
NO_CGN = {}
args.original = False

generate_dataset("colored_MNIST", 1, args.N, True)
NO_CGN["colored_MNIST"] = train_("colored_MNIST_counterfactual", args)

generate_dataset("double_colored_MNIST", 1, args.N, True)
NO_CGN["double_colored_MNIST"] = train_("double_colored_MNIST_counterfactual", args)

generate_dataset("wildlife_MNIST", 1, args.N, True)
NO_CGN["wildlife_MNIST"] = train_("wildlife_MNIST_counterfactual", args)

table2[row] = NO_CGN

with open(json_file, 'w') as f:
    json.dump(table2, f)

# Train on original + GAN
row = "original + GAN"
O_GAN = {}
args.original = True

generate_dataset("colored_MNIST", 1, args.N, False)
O_GAN["colored_MNIST"] = train_("colored_MNIST_counterfactual", args)

generate_dataset("double_colored_MNIST", 1, args.N, False)
O_GAN["double_colored_MNIST"] = train_("double_colored_MNIST_counterfactual", args)

generate_dataset("wildlife_MNIST", 1, args.N, False)
O_GAN["wildlife_MNIST"] = train_("wildlife_MNIST_counterfactual", args)

table2[row] = O_GAN

with open(json_file, 'w') as f:
    json.dump(table2, f)

# Train on GAN
row = "GAN"
NO_GAN = {}
args.original = False

generate_dataset("colored_MNIST", 1, args.N, False)
NO_GAN["colored_MNIST"] = train_("colored_MNIST_counterfactual", args)

generate_dataset("double_colored_MNIST", 1, args.N, False)
NO_GAN["double_colored_MNIST"] = train_("double_colored_MNIST_counterfactual", args)

generate_dataset("wildlife_MNIST", 1, args.N, False)
NO_GAN["wildlife_MNIST"] = train_("wildlife_MNIST_counterfactual", args)

table2[row] = NO_GAN

with open(json_file, 'w') as f:
    json.dump(table2, f)

# Train on original
row = "original"
original = {}

original["colored_MNIST"] = train_("colored_MNIST", args)

original["double_colored_MNIST"] = train_("double_colored_MNIST", args)

original["wildlife_MNIST"] = train_("wildlife_MNIST", args)

table2[row] = original

with open(json_file, 'w') as f:
    json.dump(table2, f)
