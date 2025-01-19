import time
import os
import copy
import torch
import torchvision
import matplotlib.pyplot as plt
import argparse

from model_config import ModelConfig



class TrainingPipeline:
    def __init__(self, model_name):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = ModelConfig[model_name]
        self.model = self.config.model.to(self.device)
        self.processer = self.config.processer


    def prepare_dataloader(self):
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=self.processer)
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [40000, 10000])
        
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=True, transform=self.processer)
        
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128,
                                                shuffle=True, num_workers=2)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=128,
                                                shuffle=True, num_workers=2)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=128,
                                                shuffle=False, num_workers=2)
                                                
        
        return { 'train': train_dataloader, 'val': val_dataloader, 'test':test_dataloader }


    def train_model(self):
        since = time.time()
        dataloaders = self.prepare_dataloader()

        val_acc_history = []
        loss_acc_history = []

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()

            print('Epoch {}/{}'.format(epoch, self.config.num_epochs - 1))
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train() 
                else:
                    self.model.eval()  

                running_loss = 0.0
                running_corrects = 0
                
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    self.config.optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        if self.config.is_inception and phase == 'train':
                            outputs, aux_outputs = self.model(inputs)
                            loss1 = self.config.criterion(outputs, labels)
                            loss2 = self.config.criterion(aux_outputs, labels)
                            loss = loss1 + 0.4 * loss2
                        else:
                            outputs = self.model(inputs)
                            loss = self.config.criterion(outputs[0], labels)

                        if isinstance(outputs, tuple):
                            outputs = outputs[0]

                        _, preds = torch.max(outputs, 1)

                        if phase == 'train':
                            loss.backward()
                            self.config.optimizer.step()
                    
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
                epoch_end = time.time()
                
                elapsed_epoch = epoch_end - epoch_start

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                print("Epoch time taken: ", elapsed_epoch)

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                    # if not os.path.exists(self.config.weights_name):
                    #     os.mkdir(self.config.weights_name)
                    torch.save(self.model.state_dict(), f"{self.config.model_path}/{self.config.weights_name}.pth")
                if phase == 'val':
                    val_acc_history.append(epoch_acc)
                if phase == 'train':
                    loss_acc_history.append(epoch_loss)
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))
        self.model.load_state_dict(best_model_wts)

        val_acc_history = [t.cpu().item() for t in val_acc_history]
        loss_acc_history = [t.cpu().item() for t in loss_acc_history]
        # Plot loss per epoch
        plt.plot(loss_acc_history, label='Validation')
        plt.title('Loss per epoch')
        plt.legend()
        plt.savefig(f'{self.config.plot_save_dir}/loss_per_epoch.png') 
        # plt.show()  

        # Plot accuracy per epoch
        plt.plot(val_acc_history, label='Validation')
        plt.title('Accuracy per epoch')
        plt.legend()
        plt.savefig(f'{self.config.plot_save_dir}/accuracy_per_epoch.png')
        # plt.show()

        return self.model, val_acc_history, loss_acc_history
    

    def evaluate_model(self, model, dataloaders):
        # Set the model to evaluation mode
        model.eval()
        
        # dataloaders = self.prepare_dataloader()
        test_dataloader = dataloaders['test']
        
        running_corrects = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                _, preds = torch.max(outputs, 1)

                # Accumulate the number of correct predictions
                running_corrects += torch.sum(preds == labels.data)
                total_samples += labels.size(0)

        # Calculate accuracy
        accuracy = running_corrects.double() / total_samples
        print(f"Test Accuracy: {accuracy:.4f}")

        return accuracy.item()
    


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument('model_name', choices=['GOOGLENET', 'ALEXNETLRN', 'ALEXNETWITHOUTLRN'], 
                        help="Specify the model to use for training (GOOGLENET, ALEXNETLRN, ALEXNETWITHOUTLRN)")
    return parser.parse_args()

        

if __name__ == "__main__":
    args = parse_args()

    if args.model_name in ModelConfig.__members__:
        pipeline = TrainingPipeline(args.model_name)
        pipeline.train_model()
    else:
        print("Model not available. Avialable model names are (GOOGLENET, ALEXNETLRN and ALEXNETWITHOUTLRN.")