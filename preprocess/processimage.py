import os
import pandas as pd
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

datasets_path = 'Clothing_Shoes_and_Jewelry/'

batch_size = 512
datasets_name = 'Clothes'
image_path = 'dict/' + datasets_path + 'Clothes_images/'
asin_list_path = 'dict/' + datasets_path + 'asinlist.npy'
asin_list = np.load(asin_list_path, allow_pickle=True)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model = models.resnet50(pretrained=True)
model = model.eval() 

img_embeddings = []
img_name_list_good = []

num_img = len(asin_list)
epoch = num_img // batch_size
not_RGB = 0
error_img_name = []

for i in tqdm(range(epoch)):
    img_temp_list = asin_list[i * batch_size:(i + 1) * batch_size]
    input_list = []
    for img_temp in tqdm(img_temp_list):
        image_temp_path = os.path.join(image_path, f"{img_temp}.jpg")
        try:
            input_image = Image.open(image_temp_path)
            if input_image.mode != 'RGB':
                not_RGB += 1
                input_image = input_image.convert('RGB')
            input_tensor = preprocess(input_image)
        except Exception as e:
            print(f'Error processing image: {img_temp}, {e}')
            error_img_name.append(img_temp)
            continue
        img_name_list_good.append(img_temp)
        input_list.append(input_tensor.tolist())

    input_batch = torch.Tensor(input_list)  # [batch_size, 3, 224, 224]
    with torch.no_grad():
        features = model(input_batch)
    img_embeddings.extend(features.numpy())

if num_img % batch_size != 0:
    img_temp_list = asin_list[epoch * batch_size:]
    input_list = []
    for img_temp in tqdm(img_temp_list):
        image_temp_path = os.path.join(image_path, f"{img_temp}.jpg")
        try:
            input_image = Image.open(image_temp_path)
            if input_image.mode != 'RGB':
                not_RGB += 1
                input_image = input_image.convert('RGB')
            input_tensor = preprocess(input_image)
        except Exception as e:
            print(f'Error processing image: {img_temp}, {e}')
            error_img_name.append(img_temp)
            continue
        img_name_list_good.append(img_temp)
        input_list.append(input_tensor.tolist())

    input_batch = torch.Tensor(input_list)
    with torch.no_grad():
        features = model(input_batch)
    img_embeddings.extend(features.numpy())

output_df = pd.DataFrame(img_embeddings, index=img_name_list_good)
output_df.to_csv(datasets_name + '/img_features.csv')

print(f"Image features saved to {datasets_name + '/img_features.csv'}")
print('total image: ', str(len(img_embeddings)))
