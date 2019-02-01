import torch
from Models import ConvNet
from Loading import load_test_data
from sklearn.metrics import classification_report

from tqdm import tqdm

model = ConvNet(10)

state_dict = torch.load("C:/Users/Home/Desktop/final/ConvNet_55.model")
model.load_state_dict(state_dict)

data_size = 272
test_batch_size = 64
size = 64

test_path = "C:/Users/Home/Desktop/Data science/Girls in tech/Neural networks/Exam/monkey_recognition/data/validation"

data_loader = load_test_data(test_path, 128, size, shuffle=False)

def test():
    model.eval()
    acc = 0
    y_hat = []
    y_true = []
    for X, y in tqdm(data_loader):
        out = model(X)
        
        predictions = torch.argmax(out, 1)
        acc += torch.sum(predictions == y).item()
        y_hat.append(predictions)
        y_true.append(y)
        
    y_hat = torch.cat(y_hat)
    y_true = torch.cat(y_true)
    acc = acc/data_size
    print(acc)
    print(classification_report(y_hat, y_true))

if __name__ == "__main__":
    test()