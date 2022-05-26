import os
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import albumentations as A
from albumentations.pytorch import ToTensorV2

############## Augmentations ###############
both_transform = A.Compose(
    [A.Resize(width=256, height=256),], additional_targets={"image0": "image"},
)

transform_only_input = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.2),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

transform_only_mask = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

class Satellite2Map_Data(Dataset):
    def __init__(self,root):
        self.root = root
        list_files = os.listdir(self.root)
        #### Removing '.ipynb_checkpoints' from the list
        list_files.remove('.ipynb_checkpoints')
        self.n_samples = list_files
        
            
    
    def __len__(self):
        return len(self.n_samples)
    
    def __getitem__(self,idx):
        try:
            if torch.is_tensor(idx):
                idx = idx.tolist()
            image_name = self.n_samples[idx]
            #print(self.n_samples)
            image_path = os.path.join(self.root,image_name)
            image = np.asarray(Image.open(image_path).convert('RGB'))
            height, width,_ = image.shape
            width_cutoff = width // 2
            satellite_image = image[:, :width_cutoff,:]
            map_image = image[:, width_cutoff:,:]

            augmentations = both_transform(image=satellite_image, image0=map_image)
            input_image = augmentations["image"]
            target_image = augmentations["image0"]

            satellite_image = transform_only_input(image=input_image)["image"]
            map_image = transform_only_mask(image=target_image)["image"]
            # PIL_image = Image.fromarray(numpy_image.astype('uint8'), 'RGB')
            # satellite_image = Image.fromarray(satellite_image.astype('uint8'),'RGB')
            # map_image = Image.fromarray(map_image.astype('uint8'),'RGB')

            # if self.transform!=None:
            #     satellite_image = self.transform(satellite_image)
            #     map_image = self.transform(map_image)

            return (satellite_image, map_image)
        except:
            if torch.is_tensor(idx):
                idx = idx.tolist()
            image_name = self.n_samples[idx]
            #print(self.n_samples)
            image_path = os.path.join(self.root,image_name)
            print(image_path)
            pass
    
    
            
if __name__=="__main__":
    dataset = Satellite2Map_Data("./maps/train")
    loader = DataLoader(dataset, batch_size=5)
    for x,y in loader:
        print("X Shape :-",x.shape)
        print("Y Shape :-",y.shape)
        save_image(x,"satellite.png")
        save_image(y,"map.png")
        break            