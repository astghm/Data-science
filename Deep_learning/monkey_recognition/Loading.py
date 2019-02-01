from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

def load_train_data(train_path, train_batch_size, size, shuffle = True):
    transformers = transforms.Compose([
                    transforms.Resize((size, size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomGrayscale(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    set_data = datasets.ImageFolder(root = train_path, transform = transformers)
    return DataLoader(set_data, batch_size = train_batch_size, shuffle = shuffle, num_workers = 1)

def load_test_data(test_path, test_batch_size, size, shuffle = True):
    transformers = transforms.Compose([transforms.Resize((size, size)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    set_data = datasets.ImageFolder(root = test_path, transform = transformers)
    return DataLoader(set_data, batch_size = test_batch_size, shuffle = shuffle, num_workers = 1)