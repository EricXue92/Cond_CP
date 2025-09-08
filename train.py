import torch
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    # [0.005, 0.01]
    # 0.005
    parser.add_argument("--learning_rate", type=float, default=0.05, help="Learning rate") # breast 5e-3  #
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train for")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size to use for training") # 128
    parser.add_argument("--alpha", type=float, default=0.01, help="Conformal Rate") #####  0.05 or 0.01
    parser.add_argument("--dataset", default="CIFAR10", choices=["CIFAR100", "Alzheimer",'CIFAR10', "SVHN", "CIFAR100", "Colorectal"])
    parser.add_argument("--OOD", default="SVHN", choices=["Brain_tumors", "Alzheimer", 'CIFAR10', 'CIFAR100', "SVHN", "Colorectal", "Breast"])
    parser.add_argument("--n_inducing_points", type=int, default=15, help="Number of inducing points") # 40
    parser.add_argument("--beta", type=float, default=0.1, help="Weight for conformal training loss")
    parser.add_argument("--temperature", type=float, default=0.01, help="Temperature for conformal training loss")
    parser.add_argument("--snn", action="store_true", help="Use standard NN or not")
    parser.add_argument("--sngp", action="store_true", help="Use SNGP or not")
    parser.add_argument("--snipgp", action="store_false", help="Use SNIPGP or not")
    parser.add_argument("--conformal_training", action="store_false", help="conformal training or not")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay") # 1e-4,   (5e-4 for CIFAR10) (5e-4 for sngp breast)
    parser.add_argument("--kernel", default="RBF", choices=["RBF", "RQ", "Matern12", "Matern32", "Matern52"], help="Pick a kernel",)
    parser.add_argument("--no_spectral_conv", action="store_true",  dest="spectral_conv", help="Don't use spectral normalization on the convolutions",)
    parser.add_argument( "--adaptive_conformal", action="store_true", help="adaptive conformal")
    parser.add_argument("--no_spectral_bn", action="store_true", dest="spectral_bn", help="Don't use spectral normalization on the batch normalization layers",)
    parser.add_argument("--coeff", type=float, default=3, help="Spectral normalization coefficient") # 3
    parser.add_argument("--n_power_iterations", default=1, type=int, help="Number of power iterations")
    parser.add_argument("--output_dir", default="./default", type=str, help="Specify output directory")
    parser.add_argument("--size_loss_form", default="identity", type=str, help="identity or log")
    parser.add_argument("--spec_norm_replace_list", nargs='+', default=["Linear", "Conv2D"], type=str, help="List of specifications to replace" )
    parser.add_argument("--spectral_normalization", action="store_false", help="Use spectral normalization or not")
    args = parser.parse_args()
    if sum([args.sngp, args.snipgp, args.snn]) != 1:
        parser.error("Exactly one of --snn, --sngp or --snipgp must be set.")
    return args

if __name__ == "__main__":
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [1, 23, 42, 202, 2024]


