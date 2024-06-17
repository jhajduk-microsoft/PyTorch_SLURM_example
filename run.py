import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler


# Define transformations
transform = transforms.Compose([
    transforms.Resize(224),  # ResNet requires the input to be 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization parameters from ImageNet
])

# Initialize the distributed environment
dist.init_process_group(backend='nccl')

# Load the CIFAR10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Create distributed samplers
train_sampler = DistributedSampler(trainset)
test_sampler = DistributedSampler(testset)

# Create data loaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False, sampler=train_sampler)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, sampler=test_sampler)

# Load the pre-trained ResNet model
net = torchvision.models.resnet50(pretrained=True)

# Replace the last layer to match the number of classes in CIFAR10
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 10)

# Move the model to GPU and make it distributed
net = net.to(device)
net = DistributedDataParallel(net)

# Define the criterion and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Train the network
for epoch in range(10):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')