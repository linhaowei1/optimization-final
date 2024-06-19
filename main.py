import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
from torchvision import datasets, transforms
from tqdm import tqdm
import wandb

from optimizers import DAdaptSGD, ProdigySGD
from model import WideResNet  # Using a pretrained model for simplicity

@torch.no_grad()
def evaluate(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            test_loss += criterion(outputs, labels).item()  # sum up batch loss
            pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(labels.view_as(pred)).sum().item()
            total += inputs.shape[0]

    test_loss /= total
    accuracy = 100. * correct / total
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.0f}%)\n')
    return test_loss, accuracy

def main(args):

    wandb.init(project='optim-final', name=f'{args.optimizer}_seed_{args.seed}')
    # Setting the random seed for reproducibility
    wandb.log({'optimizer': args.optimizer, 'seed': args.seed, 'lr': args.lr})
    torch.manual_seed(args.seed)

    # CIFAR10 Data transformations
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Loading the dataset
    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=8)
    
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=8)
    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WideResNet(depth=16, widen_factor=8, dropout_rate=0, num_classes=10).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)
    elif args.optimizer.startswith('dadaptsgd'):
        optimizer = DAdaptSGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001, log_every=200)
    elif args.optimizer.startswith('prodigysgd'):
        optimizer = ProdigySGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001, log_every=200)
    
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 225], gamma=0.1)

    training_loss = []
    evaluation_acc = []
    # Training the model
    for epoch in tqdm(range(300)):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 199:
                print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 200:.3f}')
                training_loss.append(running_loss / 200)
                wandb.log({'training_loss': running_loss / 200, 'optimizer_d': optimizer.param_groups[0]['d']})
                running_loss = 0.0
        
        scheduler.step()
        print(f'End of Epoch {epoch + 1}, Learning Rate: {scheduler.get_last_lr()}')

        test_loss, accuracy = evaluate(model, device, test_loader, criterion)
        evaluation_acc.append(accuracy)
        wandb.log({'test_loss': test_loss, 'accuracy': accuracy})

    print('Finished Training')

    save_dir = os.path.join('results', args.optimizer, f'seed_{args.seed}')
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, 'training_loss.json'), 'w') as f:
        json.dump(training_loss, f)
    with open(os.path.join(save_dir, 'evaluation_acc.json'), 'w') as f:
        json.dump(evaluation_acc, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Wide ResNet Training')
    parser.add_argument('--optimizer', type=str, default='sgd', help='Optimizer to use')
    parser.add_argument('--lr', type=float, default=0.1, help='Optimizer to use')
    parser.add_argument('--seed', type=int, default=10, help='Random seed for reproducibility')
    args = parser.parse_args()
    main(args)
