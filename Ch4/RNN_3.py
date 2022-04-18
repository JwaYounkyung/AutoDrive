# Word RNN
import torch
import torch.nn as nn
import torch.optim as optim

sentence = "Repeat is the best medicine for memory".split()
vocab = list(set(sentence))
print(vocab)

word2index = {tkn: i for i, tkn in enumerate(vocab, 1)}
word2index['<unk>']=0
print(word2index)
print(word2index['memory'])

index2word = {v: k for k, v in word2index.items()}
print(index2word)
print(index2word[2])

def build_data(sentence, word2index):
    encoded = [word2index[token] for token in sentence]
    input_seq, label_seq = encoded[:-1], encoded[1:]
    input_seq = torch.LongTensor(input_seq).unsqueeze(0) # 첫단어 ~ 여섯
    label_seq = torch.LongTensor(label_seq).unsqueeze(0) # 두번째 ~ 일곱
    return input_seq, label_seq

X, Y = build_data(sentence, word2index)
print(X)
print(Y)

class Net(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size, batch_first=True):
        super(Net, self).__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size,
                                            embedding_dim=input_size)
        self.rnn_layer = nn.RNN(input_size, hidden_size,
                                batch_first=batch_first)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        output = self.embedding_layer(x)
        output, hidden = self.rnn_layer(output)
        output = self.linear(output)
        return output.view(-1, output.size(2))

vocab_size = len(word2index)
input_size = 5
hidden_size = 20

model = Net(vocab_size, input_size, hidden_size, batch_first=True)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters())

output = model(X)
print(output)
print(output.shape)

decode = lambda y: [index2word.get(x) for x in y]

# 맨뒤에 나오는 단어 예측하는게 목표
for step in range(201):
    optimizer.zero_grad()
    output = model(X)
    loss = loss_function(output, Y.view(-1))
    loss.backward()
    optimizer.step()
    if step % 40 == 0:
        print("[{:02d}/201] {:.4f} ".format(step+1, loss))
        pred = output.softmax(-1).argmax(-1).tolist()
        print(" ".join(["Repeat"] + decode(pred)))
        print()