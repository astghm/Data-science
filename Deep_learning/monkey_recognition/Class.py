import torch
from Models import ConvNet
from PIL import Image
from torchvision import transforms
import numpy as np

class monkey_classifier():
    def __init__(self, model_path):
        self._load_model(model_path)
        self._prepare_transform()
        self.i2class = ["mantled_howler", "patas_monkey", "bald_uakari", "japanese_macaque", "pygmy_marmoset",
                       "white_headed_capuchin", "silvery_marmoset", "common_squirrel_monkey", "black_headed_night_monkey",
                       "nilgiri_langur"]

    def _load_model(self, model_path):
        model = ConvNet(10) 
        state_dict = torch.load(model_path) 
        model.load_state_dict(state_dict)
        model.eval()
        self.model = model

    def _prepare_transform(self):
        self.transformer = transforms.Compose([
                    transforms.Resize((64, 64)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


    def predict(self, path_img):
        img = Image.open(path_img)
        img = self.transformer(img)
        img = img.view(1, 3, 64, 64)
        print(img.shape)
        out = self.model(img)  
        prediction = torch.argmax(out, 1).item()

        return self.i2class[prediction]
    
    def predict_proba(self, path_img):
        img = Image.open(path_img)
        img = self.transformer(img)
        img = img.view(1, 3, 64, 64)
        out = self.model(img)
        softmax = torch.nn.Softmax()
        pred_prob = softmax(out).detach().numpy().reshape(10,)
        prob_dict = {}
        for i in range(len(self.i2class)):
            prob_dict[self.i2class[i]] = pred_prob[i]

        return prob_dict