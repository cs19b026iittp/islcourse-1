import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.nn.functional as F
from torchvision.transforms import ToTensor
  
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

loss_fn = nn.CrossEntropyLoss()

# neural network
class cs19b032NN(nn.Module):
 def __init__(self, img_size):
        super(cs19b032NN, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(img_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

 def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
# sample invocation torch.hub.load(myrepo,'get_model',train_data_loader=train_data_loader,n_epochs=5, force_reload=True)
def get_model(train_data_loader=None, n_epochs=1):
  
  dataiter = iter(train_data_loader)
  img, _ = dataiter.next()
  
  model = cs19b032NN(img.shape[2]*img.shape[3]).to(device)
  optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

  size = len(train_data_loader.dataset)
  model.train()
  for epoch in range(1,n_epochs+1):
    print("Epoch ", epoch);
    for batch, (X, y) in enumerate(train_data_loader):
      X, y = X.to(device), y.to(device)

      # Compute prediction error
      pred = model(X)
      loss = loss_fn(pred, y)

      # Backpropagation
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if batch % 100 == 0:
          loss, current = loss.item(), batch * len(X)
          print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

  
  print ('Returning model... (rollnumber: cs19b032)')
  
  return model

# sample invocation torch.hub.load(myrepo,'get_model_advanced',train_data_loader=train_data_loader,n_epochs=5, force_reload=True)
def get_model_advanced(train_data_loader=None, n_epochs=10,lr=1e-4,config=None):
  model = get_model(train_data_loader, n_epochs)

  # write your code here as per instructions
  # ... your code ...
  # ... your code ...
  # ... and so on ...
  # Use softmax and cross entropy loss functions
  # set model variable to proper object, make use of train_data
  
  # In addition,
  # Refer to config dict, where learning rate is given, 
  # List of (in_channels, out_channels, kernel_size, stride=1, padding='same')  are specified
  # Example, config = [(1,10,(3,3),1,'same'), (10,3,(5,5),1,'same'), (3,1,(7,7),1,'same')], it can have any number of elements
  # You need to create 2d convoution layers as per specification above in each element
  # You need to add a proper fully connected layer as the last layer
  
  # HINT: You can print sizes of tensors to get an idea of the size of the fc layer required
  # HINT: Flatten function can also be used if required
  return model
  
  
  print ('Returning model... (rollnumber: xx)')
  
  return model

# sample invocation torch.hub.load(myrepo,'test_model',model1=model,test_data_loader=test_data_loader,force_reload=True)
def test_model(model1=None, test_data_loader=None):

  accuracy_val, precision_val, recall_val, f1score_val = 0, 0, 0, 0
  
  size = len(test_data_loader.dataset)
  num_batches = len(test_data_loader)
  model1.eval()
  test_loss, correct = 0, 0
  

  y_true = []
  y_pred = []

  with torch.no_grad():
      for X, y in test_data_loader:
          y_true.append(y)
          X, y = X.to(device), y.to(device)
          pred = model1(X)
          test_loss += loss_fn(pred, y).item()
          y_pred.append(pred.argmax(1))
          correct += (pred.argmax(1) == y).type(torch.float).sum().item()
  test_loss /= num_batches
  correct /= size


  accuracy_val = correct

  print ('Returning metrics... (rollnumber: cs19b032)')
  
  return accuracy_val, precision_val, recall_val, f1score_val

