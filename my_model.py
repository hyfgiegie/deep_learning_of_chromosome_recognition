from typing import Optional, Union

import torch
from torch import nn, device
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear, ReLU, Softmax, Dropout
from torch.nn.modules.module import T
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import fetch_data


class MyModule(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.module = Sequential(
            Conv2d(1, 64, 3, 1, 1),
            ReLU(),
            # Conv2d(64, 64, 3, 1, 1),
            # ReLU(),
            MaxPool2d(2, 2),

            Conv2d(64, 128, 3, 1, 1),
            ReLU(),
            # Conv2d(64, 64, 3, 1, 1),
            # ReLU(),
            MaxPool2d(2, 2),

            Conv2d(128, 256, 3, 1, 1),
            ReLU(),
            # Conv2d(128, 128, 3, 1, 1),
            # ReLU(),
            # Conv2d(128, 128, 3, 1, 1),
            # ReLU(),
            MaxPool2d(2, 2),

            Conv2d(256, 512, 3, 1, 1),

            ReLU(),
            # Conv2d(256, 256, 3, 1, 1),
            # ReLU(),
            # Conv2d(256, 256, 3, 1, 1),
            # ReLU(),
            MaxPool2d(2, 2),

            Conv2d(512, 512, 3, 1, 1),
            ReLU(),
            # Conv2d(256, 256, 3, 1, 1),
            # ReLU(),
            # Conv2d(256, 256, 3, 1, 1),
            # ReLU(),
            MaxPool2d(2, 2),

            Flatten(),
            Linear(25088, 4096),
            ReLU(),
            Dropout(0.5),
            Linear(4096, 1000),
            ReLU(),
            Dropout(0.5),
            Linear(1000, 24),

            # Linear(4096, 1000),
            # ReLU(),
            # Linear(1000, 24),
            # ReLU(),

            # Softmax(dim=1)
        )

    def forward(self, x):
        x = self.module(x)
        return x


writer = SummaryWriter("logs")
train_dataloader = fetch_data.train_dataloader
test_dataloader = fetch_data.test_dataloader
# module = MyModule()
# step = 0
# for data in vision_test.dataloader:
#     imgs, labels = data
#     output = module(imgs)
#     writer.add_images("input", imgs, step)
#     output = torch.reshape(output, [-1, 3, 11, 11])
#     writer.add_images("output", output, step)
#     step = step + 1
# writer.close()

# model = MyModule().cuda()
# cro_loss = nn.CrossEntropyLoss().cuda()
# optim = torch.optim.Adam(model.parameters(), 0.01)
# for epoch in range(64):
#     running_loss = 0.0
#     for data in fetch_data.train_dataloader:
#         optim.zero_grad()
#         imgs, labels = data
#         imgs = imgs.cuda()
#         labels = labels.cuda()
#         output = model(imgs)
#         loss = cro_loss(output, labels)
#         loss.backward()
#         optim.step()
#         running_loss += loss
#     print(running_loss)

vgg_model = MyModule().cuda()
cro_loss = torch.nn.CrossEntropyLoss().cuda()

optim = torch.optim.SGD(vgg_model.parameters(), 0.01)

epoch = 64
for i in range(epoch):
    cnt = 0
    running_loss = torch.tensor(0.0).cuda()
    for data in train_dataloader:
        imgs, labels = data
        imgs = imgs.cuda()
        labels = labels.cuda()
        output = vgg_model(imgs)
        cnt += 1
        loss = cro_loss(output, labels)
        optim.zero_grad()
        loss.backward()
        optim.step()
        running_loss += loss
    print("第{}轮训练结果：Loss = {}".format(i + 1, running_loss.item()))
    writer.add_scalar("total_loss", running_loss, i + 1)
    torch.save(vgg_model.state_dict(), "./train_result/vgg16_method{}.pth".format(i + 1))

    with torch.no_grad():
        sum_right_num = 0
        for data in test_dataloader:
            imgs, labels = data
            imgs = imgs.cuda()
            labels = labels.cuda()
            outputs = vgg_model(imgs)
            preds = outputs.argmax(1)
            right_num = (preds == labels).sum().item()
            sum_right_num += right_num
        accuracy = sum_right_num / len(test_dataloader.dataset)
        print("第{}轮训练结果：Accuracy = {}".format(i + 1, accuracy))
        writer.add_scalar("test_accuracy", accuracy, i + 1)


writer.close()
