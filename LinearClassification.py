import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from resnet_18 import get_ResNet18
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from tqdm import tqdm


def linear_classification(pretrain_params, pretrained_path, epochs, batch_size, out_dimen, learning_rate):
    # 加载cifar-100数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # 使用CIFAR-100数据集的均值和标准差
    ])
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    writer = SummaryWriter(f'/runs/Linear Classification Protocol/{pretrain_params[0]}_{pretrain_params[1]}_{pretrain_params[2]}')

    base_model = get_ResNet18(out_dimension=out_dimen).cuda()
    base_model.load_state_dict(torch.load(pretrained_path))


    for param in base_model.feature_extractor.parameters():
        param.requires_grad = False

    num_features = 512
    base_model.projection_head = nn.Linear(num_features, 100).cuda()
    model = base_model
    for name, param in base_model.named_parameters():
        if param.requires_grad:
            print(f"{name} is not frozen")
        else:
            print(f"{name} is frozen")
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss, total_correct, total_images, total_batches = 0, 0, 0, 0
        for images, labels in train_loader:
            total_batches += 1
            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(images)
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

        model.eval()
        total_correct, total_images = 0, 0
        with torch.no_grad():   # 不计算参数梯度
            for images, labels in test_loader:
                images, labels = images.cuda(), labels.cuda()
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total_images += labels.size(0)
        test_accuracy = total_correct / total_images
        writer.add_scalar('Accuracy/test', test_accuracy, epoch)
        print(f'Test Accuracy: {test_accuracy}')
        if epoch == epochs - 1:
            torch.save(model, f'/models/Linear Classification Protocol_epoch_{epoch}_{pretrain_params[0]}_{pretrain_params[1]}_{pretrain_params[2]}.pth')
            torch.save(model.state_dict(), f'/models/Linear Classification Protocol_epoch_{epoch}_{pretrain_params[0]}_{pretrain_params[1]}_{pretrain_params[2]}.pth')
    with open("cifar10_records.txt", "a+", encoding="utf-8") as f:
        f.write(str(pretrain_params) + str(test_accuracy) + "\n")
    writer.close()

if __name__ == "__main__":
    ori_params = [0.01, 128, 0.5]   # 设置超参数
    linear_classification(ori_params, f'/runs/cifar-10/resnet18_epoch_2_{0.01}_{256}_{0.5}.pth', 5, 256, 128, 0.01)

