import torch, sys, argparse
from pthflops import count_ops

#Local imports
sys.path.append(r"../training")
from models import FashionNet


if __name__ == "__main__":
    '''
    Main function, used to parse the arguments and call the main function
    '''
    parser = argparse.ArgumentParser(description="Evaluation on a data")
    parser.add_argument('-model_name', '--model_name', type= str, help= 'model name', default="resnet-50")
    args = parser.parse_args()

    model = FashionNet(model_name = args.model_name, num_classes = 22, dropout = 0.5, freeze_backbone = False)

    inp = torch.rand(1,3,256,256)

    # Count the number of FLOPs
    count_ops(model, inp)

    #Total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
