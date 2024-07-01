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
from torchvision.transforms import ToPILImage, ToTensor


# 使用SimCLR在CIFAR-10数据集上训练ResNet-18
def simclr(epochs, learning_rate, batchsize=128, out_dimen=128, save_steps=5, step_size=20, gamma=0.5):
    writer = SummaryWriter(f'/runs/cifar-10/resnet_18_{learning_rate}_{batchsize}_{gamma}')

    # 数据集准备
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    augmentation_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=32),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
        ToTensor()
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=base_transform)
    data = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)

    model = get_ResNet18(out_dimension=out_dimen).cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    to_pil_image = ToPILImage()

    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss = 0
        num_batches = 0
        for batch in data:
            images, _ = batch
            images = images.cuda()

            # 创建图像的两个不同增强版本
            images_aug1 = torch.stack([augmentation_transform(to_pil_image(img.cpu())) for img in images])
            images_aug2 = torch.stack([augmentation_transform(to_pil_image(img.cpu())) for img in images])

            images_aug1 = images_aug1.cuda()
            images_aug2 = images_aug2.cuda()

            optimizer.zero_grad()
            z_i = model(images_aug1)
            z_j = model(images_aug2)

            loss = nt_xent_loss(z_i, z_j)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        average_loss = total_loss / num_batches
        print(f'第 {epoch} 轮，损失：{average_loss}')
        writer.add_scalar('Loss/train', average_loss, epoch)

        if epoch == epochs - 1 or (epoch + 1) % save_steps == 0:
            torch.save(model.state_dict(),
                       f'/runs/cifar-10/resnet18_epoch_{epoch + 1}_{learning_rate}_{batchsize}_{gamma}.pth')
            print('模型已保存，轮次', epoch + 1)

        scheduler.step()


# 示例使用
if __name__ == "__main__":
    simclr(epochs=50, learning_rate=0.01, batchsize=256, out_dimen=128, save_steps=10, step_size=20, gamma=0.5)
