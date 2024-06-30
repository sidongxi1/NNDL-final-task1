import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from resnet_18 import get_ResNet18
from loss_function import nt_xent_loss
from tqdm import tqdm

# 使用SimCLR在CIFAR-10数据集上训练ResNet-18
def simclr(epochs, learning_rate, batchsize=128, out_dimen=128, save_steps=5, step_size=20, gamma=0.5):
    writer = SummaryWriter(f'/runs/cifar-10/resnet_18_{learning_rate}_{batchsize}_{gamma}')

    # 数据集准备
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # 使用CIFAR-10数据集的均值和标准差
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    data = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)

    model = get_ResNet18(out_dimension=out_dimen).cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss = 0
        num_batches = 0
        for batch in data:
            images, _ = batch
            images = images.cuda()
            optimizer.zero_grad()
            z_i = model(images)
            z_j = model(images)

            loss = nt_xent_loss(z_i, z_j)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        average_loss = total_loss / num_batches
        print(f'Epoch {epoch}, Loss: {average_loss}')
        writer.add_scalar('Loss/train', average_loss, epoch)
        
        if epoch == epochs - 1:
            torch.save(model.state_dict(), f'/model/cifar-10/resnet18_epoch_{epoch+1}_{learning_rate}_{batchsize}_{gamma}.pth')
            print('saved')
        
        scheduler.step()

    writer.close()


if __name__ == "__main__":
    simclr(epochs=2, learning_rate=0.01, batchsize=256, out_dimen=128, save_steps=2, step_size=10, gamma=0.5)
