from enum import Enum

from googlenet.googlenet import GoogLeNet
from alexnet.alexnet_with_lrn import AlexNetLRN
from alexnet.alexnet_without_lrn import AlexNetWithoutLRN
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms




class ModelConfig(Enum):
    ALEXNETLRN = {
         "model": AlexNetLRN(),
        "criterion":nn.CrossEntropyLoss(),
        "optimizer":optim.SGD(AlexNetLRN().parameters(), lr=0.001, momentum=0.9),
        "processer":transforms.Compose([
                                            transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
        "num_epochs":10,
        "is_inception":False,
        "weights_name":"/saved_models/alexnet",
        "model_path": "saved_models/alexnet",
        "plot_save_dir":"saved_plots/alexnet"
    }
    ALEXNETWITHOUTLRN = {
        "model": AlexNetWithoutLRN(),
        "criterion":nn.CrossEntropyLoss(),
        "optimizer":optim.SGD(AlexNetWithoutLRN().parameters(), lr=0.001, momentum=0.9),
        "processer":transforms.Compose([
                                            transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
        "num_epochs":10,
        "is_inception":False,
        "weights_name":"alexnet_best",
        "model_path": "saved_models/alexnet",
        "plot_save_dir":"saved_plots/alexnet"
    }
    GOOGLENET = {
        "model": GoogLeNet(),
        "criterion":nn.CrossEntropyLoss(),
        "optimizer":optim.Adam(GoogLeNet().parameters(), lr=0.01),
        "processer":transforms.Compose([
                                            transforms.Resize(36),
                                            transforms.CenterCrop(32),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                        ]),
        "num_epochs":25,
        "is_inception":False,
        "weights_name":"googlenet_best",
        "model_path": "saved_models/googlenet",
        "plot_save_dir":"saved_plots/googlenet"
    }
    
    def __init__(self, config_dict):
        # Initialize each config value
        self.model = config_dict["model"]
        self.criterion = config_dict["criterion"]
        self.optimizer = config_dict["optimizer"]
        self.processer = config_dict["processer"]
        self.num_epochs = config_dict["num_epochs"]
        self.is_inception = config_dict["is_inception"]
        self.weights_name = config_dict["weights_name"]
        self.model_path = config_dict["model_path"]
        self.plot_save_dir = config_dict["plot_save_dir"]

    def __str__(self):
        return f"Model Configuration"


if __name__ == "__main__":
    config = ModelConfig.ALEXNET
    print(config.num_epochs)

