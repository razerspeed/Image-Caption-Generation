import pandas as pd
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

pd.set_option('display.max_colwidth', None)



import requests

server1_url = 'http://127.0.0.1:5000/foo'
server2_url = 'http://127.0.0.1:5001/foo'
# myobj = {'somekey': 'somevalue'}

def predict(image_file_name):

    # loading resizing and transforming image
    class extractImageFeatureResNetDataSet(Dataset):
        def __init__(self, data):
            self.data = data
            self.scaler = transforms.Resize([224, 224])
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
            self.to_tensor = transforms.ToTensor()
        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):

            image_name = self.data.iloc[idx]['image']
            img_loc = str(image_name)

            img = Image.open(img_loc)
            t_img = self.normalize(self.to_tensor(self.scaler(img)))

            return image_name, t_img

    image_name=image_file_name
    train_ImageDataset_ResNet = extractImageFeatureResNetDataSet(pd.DataFrame([image_name],columns=['image']))
    train_ImageDataloader_ResNet = DataLoader(train_ImageDataset_ResNet, batch_size = 1, shuffle=False)

    for image_name,image_tensor in train_ImageDataloader_ResNet:
        # print(image_name,image_tensor.shape)
        break




    server1 = requests.post(server1_url, json = {'image':image_tensor.tolist()})

    # x=req
    # uests.post(url, files={'file': image_tensor})
    # extracting data in json format
    data = server1.json()
    # print(type(data))
    # print(torch.tensor(data['image_embedding']).shape)
    #
    server2=requests.post(server2_url, json = data)

    # print(server2.json())

    prediction=server2.json()['prediction']
    # print(prediction)
    return prediction

# print(predict('1.jpg'))