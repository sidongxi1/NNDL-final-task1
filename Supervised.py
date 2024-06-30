import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torchvision.models import resnet18
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from tqdm import tqdm


def linear_evaluation(epochs, batch_size, learning_rate):
    # 加载cifar-100数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # 使用CIFAR-100数据集的均值和标准差
    ])
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    writer = SummaryWriter('/runs/supervised')

    base_model = resnet18(pretrained=False).cuda()
    # base_model.fc = nn.Linear(base_model.fc.in_features, 100).cuda()

    # for param in base_model.parameters():
    #     param.requires_grad = False

    # base_model.fc.weight.requires_grad = True
    # base_model.fc.bias.requires_grad = True

    optimizer = Adam(base_model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in tqdm(range(epochs)):
        base_model.train()
        total_loss, total_correct, total_images, total_batches = 0, 0, 0, 0
        for images, labels in train_loader:
            total_batches += 1
            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = base_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_images += labels.size(0)

        train_accuracy = total_correct / total_images
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        print(f'Epoch {epoch}: Loss {total_loss / total_batches}, Accuracy {total_correct / total_images}')

        base_model.eval()
        total_correct, total_images = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.cuda(), labels.cuda()
                outputs = base_model(images)
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total_images += labels.size(0)
        test_accuracy = total_correct / total_images
        writer.add_scalar('Accuracy/test', test_accuracy, epoch)
        print(f'Test Accuracy: {test_accuracy}')

        if epoch == epochs - 1:
            torch.save(base_model, f'/models/supervised_model_epoch_{epoch}.pth')
            torch.save(base_model.state_dict(), f'/models/supervised_state_dict_epoch_{epoch}.pth')

    writer.close()

if __name__ == "__main__":
    linear_evaluation(5, 256, 0.01)
