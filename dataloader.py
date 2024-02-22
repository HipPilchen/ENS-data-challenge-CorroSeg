import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from utils import RollTransform
from tqdm.auto import tqdm


class CorroSegDataset(Dataset):
    def __init__(self, data_dir, test = True, transform_img=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            data_dir (string): Directory with all the images (train or test).
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
         
        self.data_dir = data_dir
        self.transform_img = transform_img
        
        self.test = test
        self.masks = None
        
  
        if not os.path.exists(os.path.join(self.data_dir,'processed')):
            self.masks = pd.read_csv(os.path.join(self.data_dir,'raw','y_train.csv'))
            self.process()

        else:
            self.masks = pd.read_csv(os.path.join(self.data_dir,'processed','y_train.csv'))
     
    @property
    def raw_dir(self) -> str:
        return os.path.join(self.data_dir, 'raw/')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.data_dir, 'processed/')
    
    def process(self):
        
        taken_imgs = []
        taken_img_names = []
        
        os.mkdir(self.processed_dir) 
      
        print('Processing train images')
        for img_name in tqdm(os.listdir(os.path.join(self.raw_dir,'images_train'))):
            img = np.load(os.path.join(self.data_dir,'raw/images_train',img_name))
            new_img, dropped = self.process_img(img)
            if dropped:
                continue
            else:
                taken_imgs.append(new_img.flatten())
                taken_img_names.append(img_name[:-4])
                
        self.process_csv(taken_img_names)
        rbs = RobustScaler().fit(np.array(taken_imgs))    
        new_imgs = rbs.transform(np.array(taken_imgs))
        
        os.mkdir(os.path.join(self.processed_dir,'images_train'))
        for img_name, new_img in zip(taken_img_names, new_imgs):
            torch.save(torch.Tensor(new_img.reshape(36,36)),self.processed_dir+'/images_train/'+img_name+'.npy')
            
        # Free the memory to accelerate computations
        taken_imgs = None
        taken_img_names = None
        new_imgs = None

        
        test_imgs = []
        test_img_names = [] 
        print('Processing test images')
        for img_name in tqdm(os.listdir(os.path.join(self.raw_dir,'images_test'))):
            img = np.load(os.path.join(self.data_dir,'raw/images_test',img_name))
            new_img, dropped = self.process_img(img)
            test_imgs.append(new_img.flatten())
            test_img_names.append(img_name)
        
        new_test_imgs = rbs.transform(np.array(test_imgs))
        
        os.mkdir(os.path.join(self.processed_dir,'images_test'))
        for img_name, new_img in zip(test_img_names, new_test_imgs):
            torch.save(torch.Tensor(new_img.reshape(36,36)),os.path.join(self.processed_dir,'images_test/')+img_name)

    
    def process_csv(self, taken_img_names):
        
        df_masks = self.masks.copy()
        
        # Well 1 to remove and well 0 to rename to 1
        characters = 'well_1_'

        # Find indices of strings containing the character
        indices = [i for (i, s) in enumerate(df_masks['Unnamed: 0'].values) if characters in s]
        
        df_masks.drop(indices,axis = 0, inplace = True)
        
        rename_dict = {}
        
        old_characters = 'well_0_'
        old_indices = [i for (i, s) in enumerate(df_masks['Unnamed: 0'].values) if old_characters in s]
        
        for i in old_indices:
            img_name = df_masks['Unnamed: 0'].loc[i]
            rename_dict[img_name] = img_name.replace('well_0_', 'well_1_')
           
        df_masks['Unnamed: 0'] = df_masks['Unnamed: 0'].replace(rename_dict)
        

        df_masks_filtered = df_masks.loc[df_masks['Unnamed: 0'].isin(taken_img_names)]
     
        df_masks_filtered.to_csv(self.processed_dir + 'y_train.csv', index=False)
        self.masks = df_masks_filtered

    def process_img(self,img):
        if np.min(img) < -100:
            return img, True
        else:
            cleaned_img = img.copy()
            cleaned_img = np.nan_to_num(cleaned_img) # Replace NaN with 0
            return cleaned_img, False
        
        
    def __len__(self):
        
        if not self.test:
            return len(self.masks)
        else :
            file_names = os.listdir(os.path.join(self.raw_dir,'images_test'))
            return len(file_names)

    def __getitem__(self, idx):
        

        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        if not self.test:
            img_name = self.masks['Unnamed: 0'].iloc[idx]
            img_path = os.path.join(self.processed_dir,'images_train',img_name+'.npy')
            image = torch.load(img_path).numpy()
     
            well = int(img_name[5:7].replace('_', ''))
            
            mask = self.masks.drop(('Unnamed: 0'), axis = 1).iloc[idx].values

        else:
            file_names = os.listdir(os.path.join(self.processed_dir,'images_test'))
            img_path = os.path.join(self.processed_dir,'images_test', file_names[idx])
            well = int(file_names[idx][5:7].replace('_', ''))
            image = torch.load(img_path).numpy()
            
            # Return a dummy mask for test data
            dummy_mask = np.zeros(1296)  # Adjust the size according to your needs
            mask = dummy_mask
            
        # Convert single-channel to 3-channel RGB
        image = image[np.newaxis, :]
        image = np.tile(image, (3, 1, 1))
        mask= np.tile(mask, (3, 1, 1))
        mask = mask.reshape(3, 36, 36)

        random_states = [np.random.randint(0,100), np.random.randint(0,100)]
        
        if self.transform_img is None:
            image_tensor = torch.Tensor(image) 
            mask_tensor = torch.Tensor(image) 
        elif isinstance(self.transform_img,RollTransform):    
            image_tensor = self.transform_img(torch.Tensor(image), random_states)
            mask_tensor = self.transform_img(torch.Tensor(mask), random_states)
        else: 
            image_tensor = self.transform_img(torch.Tensor(image)) 
            mask_tensor = self.transform_img(torch.Tensor(mask))

        return image_tensor, mask_tensor, torch.Tensor([well])    
            
        # if not self.test:
        #     img_name = self.masks['Unnamed: 0'].iloc[idx]
        #     img_path = os.path.join(self.processed_dir,'images_train',img_name+'.npy')
        #     image = torch.load(img_path).numpy()
        #     well = int(img_name[5:7].replace('_', ''))
            
        #     mask = self.masks.drop(('Unnamed: 0'), axis = 1).iloc[idx].values

        # else:
        #     file_names = os.listdir(os.path.join(self.processed_dir,'images_test'))
        #     img_path = os.path.join(self.processed_dir,'images_test', file_names[idx])
        #     well = int(file_names[idx][5:7].replace('_', ''))
        #     image = torch.load(img_path).numpy()


        # if self.transform_img:
        #     image = self.transform_img(torch.Tensor(image))
            
        # if self.transform_mask:
        #     mask = self.transform_mask(torch.Tensor(mask))
            
            
        # if self.test:
        #     return torch.Tensor(image) , torch.Tensor(well)
        # else:

        #     return torch.Tensor(image), torch.Tensor(mask), torch.Tensor(well)


class CorroSeg():
    def __init__(self, data_dir, csv_file, shuffle = True,
                 batch_size = 64, valid_ratio = 0.1, transform_img=None, 
                 transform_test=None, test_params={'batch_size': 64, 'shuffle': False}):
        if transform_img is None:
            self.corroseg_dataset = CorroSegDataset(data_dir, test = False)
        else:
            self.corroseg_dataset = CorroSegDataset(data_dir, test = False, transform_img=transform_img[0])
            for i, trans_image in enumerate(transform_img):
                if i > 0:
                    self.corroseg_dataset = torch.utils.data.ConcatDataset([self.corroseg_dataset,
                                                                            CorroSegDataset(data_dir, test = False, transform_img=trans_image)])
        
        self.train_data, self.valid_data = torch.utils.data.random_split(self.corroseg_dataset, [1-valid_ratio, valid_ratio])
        self.train_dataloader = DataLoader(self.train_data, batch_size=batch_size, shuffle=shuffle)
        self.valid_dataloader = DataLoader(self.valid_data, batch_size=batch_size, shuffle=shuffle)
                
        self.test_data = CorroSegDataset(data_dir, transform_img = transform_test)
        self.test_dataloader = DataLoader(self.test_data, **test_params)
        
    def get_loaders(self):
        return self.train_dataloader, self.valid_dataloader, self.test_dataloader

    def get_datasets(self):
        return self.train_data, self.valid_data, self.test_data

