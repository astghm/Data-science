import torch
from Models import ConvNet
from Loading import load_train_data

from tqdm import tqdm


EPOCHS = 56
Learning_rate = 0.001
Batch_size = 64
L2_rate = 0

image_size = 64
data_size = 1098
num_batches = data_size//Batch_size
num_classes = 10

Train_path = "C:/Users/Home/Desktop/Data science/Girls in tech/Neural networks/Exam/monkey_recognition/data/training"


model = ConvNet(num_classes)


data_loader = load_train_data(Train_path, Batch_size, image_size)


def train():
    model.train()

    crossentropy = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=Learning_rate)

    for epoch in range(EPOCHS):
        epoch_loss = 0
        epoch_acc = 0
        for X, y in tqdm(data_loader):

            optimizer.zero_grad()

            out = model(X) 

          

            loss = crossentropy(out, y)
            loss.backward()

            optimizer.step()

            epoch_loss += loss.item() 

            predictions = torch.argmax(out, 1)
            epoch_acc += torch.sum(predictions==y).item()

        epoch_loss = epoch_loss/num_batches
        epoch_acc = epoch_acc/data_size
        print(f"Epoch {epoch}:")
        print("ACC:", epoch_acc, "LOSS:", epoch_loss)

        torch.save(model.state_dict(), f"C:/Users/Home/Desktop/final/models/ConvNet_{epoch}.model")

if __name__ == "__main__": 
    train()
