# 1 layer Char RNN example
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# input : apple, output : pple!
# this is simple example for understanding RNN
# make character set
input_str = 'apple'
label_str = 'pple!'
char_vocab = sorted(list(set(input_str+label_str)))
vocab_size = len(char_vocab)
print(char_vocab)
print ('size of character set : {}'.format(vocab_size))

# define hyperparameters
input_size = vocab_size # input size = character set size
hidden_size = 5
output_size = 5
learning_rate = 0.1

# give a character a unique integer index
char_to_index = dict((c, i) for i, c in enumerate(char_vocab)) 
print(char_to_index)

# make index_to_char to get results
index_to_char={}
for key, value in char_to_index.items():
    index_to_char[value] = key
print(index_to_char)

# map each character in the input data and label data to an integer
x_data = [char_to_index[c] for c in input_str]
y_data = [char_to_index[c] for c in label_str]
print(x_data)
print(y_data)

# add batch dimension bacause nn.RNN() basically takes a 3D tensor as input.
x_data = [x_data]
y_data = [y_data]
print(x_data)
print(y_data)

#one_hot_vector
x_one_hot = [np.eye(vocab_size)[x] for x in x_data]
print(x_one_hot)

# change input data and label data to tensor
X = torch.FloatTensor(x_one_hot)
Y = torch.LongTensor(y_data)

print('training data size : {}'.format(X.shape))
print('label data size : {}'.format(Y.shape))

# define RNN model
class Net(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size): # x_t, h_t size
        super(Net, self).__init__()
        self.rnn = torch.nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size, bias=True)

    def forward(self, x):
        x, _status = self.rnn(x)
        x = self.fc(x)
        return x

net = Net(input_size, hidden_size, output_size)
outputs = net(X)
print(outputs.shape) # 3D tensor
print(outputs.view(-1, input_size).shape)
print(Y.shape)
print(Y.view(-1).shape)

# define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), learning_rate)

net.train()

#learning
for i in range(100):
    optimizer.zero_grad()
    outputs = net(X)
    loss = criterion(outputs.view(-1, input_size), Y.view(-1))
    loss.backward()
    optimizer.step()

    result = outputs.data.numpy().argmax(axis=2)
    result_str = ''.join([index_to_char[c] for c in np.squeeze(result)])
    print(i, "loss: ", loss.item(), "prediction: ", result, "true Y: ", y_data, "prediction str: ", result_str)