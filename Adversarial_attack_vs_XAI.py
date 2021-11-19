# .py version of my colab notebook
import torch, torchvision
from torch import nn,optim
from torch.autograd import Variable as var 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.models as models
from torchsummary import summary
import numpy as np
import shap

n_batch = 64
learning_rate = 0.01
num_epochs = 1
n_print = 10

T = torchvision.transforms.ToTensor()
train_data = torchvision.datasets.MNIST('mnist_data',train=True,download=True,transform=T)
val_data = torchvision.datasets.MNIST('mnist_data',train=False,download=True,transform=T)

train_dl = torch.utils.data.DataLoader(train_data,batch_size = n_batch)
val_dl = torch.utils.data.DataLoader(val_data,batch_size = n_batch)

def imshow(img):
  img = img / 2 + 0.5 
  npimg = img.numpy() 
  plt.figure(figsize=(10,12))

  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.show()


dataiter = iter(train_dl)
images, labels = dataiter.next()
print(images.shape)
print(labels.shape)


imshow(torchvision.utils.make_grid(images))

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 20 * 20, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)


    def forward(self, x):
        # n = x.size(0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))  
        # print(x.size())
        x = x.view(-1,16 * 20 * 20)  
        # print(x.size())        
        x = F.relu(self.fc1(x))               
        x = F.relu(self.fc2(x))               
        x = self.fc3(x)                       
        return x

net = ConvNet()

summary(net, (1, 28, 28))

# training
mycnn = ConvNet()
cec = nn.CrossEntropyLoss()
optimizer = optim.Adam(mycnn.parameters(),lr = learning_rate)

def validate(model,data):
  total = 0
  correct = 0
  for i,(images,labels) in enumerate(data):
    images = var(images)
    x = model(images)
    value,pred = torch.max(x,1)
    pred = pred.data.cpu()
    total += x.size(0)
    correct += torch.sum(pred == labels)
  return correct*100./total

for e in range(num_epochs):
  for i,(images,labels) in enumerate(train_dl):
    images = images
    labels = labels
    optimizer.zero_grad()
    pred = mycnn(images)
    loss = cec(pred,labels)
    loss.backward()
    optimizer.step()
    if (i+1) % n_print == 0:
      accuracy = float(validate(mycnn,val_dl))
      print('Epoch :',e+1,'Batch :',i+1,'Loss :',float(loss.data),'Accuracy :',accuracy,'%')

#some images to generate their adversarial samples
testimg_1 = torch.unsqueeze(train_data[2][0], 0)
testimg1_GT = train_data[2][1]
testimg_2 = torch.unsqueeze(train_data[1][0], 0)
testimg2_GT = train_data[1][1]

plt.imshow(testimg_1[0][0], cmap='gray')

with torch.no_grad():
  out_1 = mycnn(torch.unsqueeze(testimg_1[0], 0))
out_1 = out_1.argmax()
out_1 = out_1.item()
print('Output for unattacked testimg_1 is {} and its label is {}'.format(out_1, testimg1_GT))

plt.imshow(testimg_2[0][0], cmap='gray')

with torch.no_grad():
  out_2 = mycnn(torch.unsqueeze(testimg_2[0], 0))
out_2 = out_2.argmax()
out_2 = out_2.item()
print('Output for unattacked testimg_2 is {} and its label is {}'.format(out_2, testimg2_GT))

def compute_gradient(func, inp, **kwargs):
    inp.requires_grad = True

    loss = func(inp, **kwargs)
    loss.backward()

    inp.requires_grad = False

    return inp.grad.data

def func(inp, net=None, target=None):
    out = net(inp)
    loss = torch.nn.functional.nll_loss(out, target=torch.LongTensor([target]))

    print(f"Loss: {loss.item()}")
    return loss

def attack(tensor, net, eps=1e-3, n_iter=50):
    new_tensor = tensor.detach().clone()

    orig_prediction = net(tensor).argmax()

    # print(f"Original prediction: {orig_prediction.item()}")

    for i in range(n_iter):
        net.zero_grad()

        grad = compute_gradient(
                func, new_tensor, net=net, target=orig_prediction.item()
                )
        new_tensor = torch.clamp(new_tensor + eps * grad.sign(), -2, 2)
        new_prediction = net(new_tensor).argmax()

        if orig_prediction != new_prediction:
            # print(f"We fooled the network after {i} iterations!")
            # print(f"New prediction: {new_prediction.item()}")
            break

    return new_tensor, orig_prediction.item(), new_prediction.item()

# Performing FGSM on test imgs
attacked_testimg_1, orig_prediction, new_prediction = attack(
            testimg_1, mycnn, eps=1e-3, n_iter=100
            )

plt.imshow(attacked_testimg_1[0][0], cmap='gray')

with torch.no_grad():
  out_1 = mycnn(torch.unsqueeze(attacked_testimg_1[0], 0))
out_1 = out_1.argmax()
out_1 = out_1.item()
print('Output for unattacked testimg_1 is {} and its label is {}'.format(out_1, testimg1_GT))

attacked_testimg_2, orig_prediction, new_prediction = attack(
            testimg_2, mycnn, eps=1e-3, n_iter=100
            )

plt.imshow(attacked_testimg_2[0][0], cmap='gray')

with torch.no_grad():
  out_2 = mycnn(torch.unsqueeze(attacked_testimg_2[0], 0))
out_2 = out_2.argmax()
out_2 = out_2.item()
print('Output for unattacked testimg_2 is {} and its label is {}'.format(out_2, testimg2_GT))

train_data = torchvision.datasets.MNIST('mnist_data',train=True,download=True,transform=T)

train_dl = torch.utils.data.DataLoader(train_data,batch_size = 128)

# images.shape

batch = next(iter(train_dl))
images, _ = batch

background = images[:100]
test_images = images[100:103]

e = shap.DeepExplainer(mycnn, background)
shap_values = e.shap_values(test_images)

shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)

#SHAP signatures of unattacked images
shap.image_plot(shap_numpy, -test_numpy)

batch = next(iter(val_dl))
images, _ = batch

images.shape

test_images = attacked_testimg_1

e = shap.DeepExplainer(mycnn, background)
shap_values = e.shap_values(test_images)

shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)

# shap signature of attacked image 1
shap.image_plot(shap_numpy, -test_numpy)

# shap signature of attacked image 2
test_images = attacked_testimg_2

e = shap.DeepExplainer(mycnn, background)
shap_values = e.shap_values(test_images)
shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)
shap.image_plot(shap_numpy, -test_numpy)

