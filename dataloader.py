import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from utils import RollTransform
from tqdm.auto import tqdm


class CorroSegDataset(Dataset):
    def __init__(self, data_dir, test=True, transform_img=None):
        """Dataset for the Corrosion Segmentation Challenge.

        Inputs:
            data_dir (string): Directory with all the images (train or test).
            test (boolean): True if the dataset is for test, False otherwise.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.data_dir = data_dir
        self.transform_img = transform_img

        self.test = test
        self.masks = None

        if not os.path.exists(os.path.join(self.data_dir, 'processed')):
            self.masks = pd.read_csv(os.path.join(
                self.data_dir, 'raw', 'y_train.csv'), index_col=0)
            self.process()

        else:
            self.masks = pd.read_csv(os.path.join(
                self.data_dir, 'processed', 'y_train.csv'), index_col=0)

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
        for img_name in tqdm(os.listdir(os.path.join(self.raw_dir, 'images_train'))):
            img = np.load(os.path.join(
                self.data_dir, 'raw/images_train', img_name))
            new_img, dropped = self.process_img(img)
            if dropped:
                continue
            else:
                taken_imgs.append(new_img.flatten())
                taken_img_names.append(img_name[:-4])

        self.process_csv(taken_img_names)
        rbs = RobustScaler().fit(np.array(taken_imgs))
        new_imgs = rbs.transform(np.array(taken_imgs))

        os.mkdir(os.path.join(self.processed_dir, 'images_train'))
        for img_name, new_img in zip(taken_img_names, new_imgs):
            torch.save(torch.Tensor(new_img.reshape(36, 36)),
                       self.processed_dir+'/images_train/'+img_name+'.npy')

        # Free the memory to accelerate computations
        taken_imgs = None
        taken_img_names = None
        new_imgs = None

        test_imgs = []
        test_img_names = []
        test_outliers_img = []
        test_outliers_img_names = []
        print('Processing test images')
        for img_name in tqdm(os.listdir(os.path.join(self.raw_dir, 'images_test'))):
            img = np.load(os.path.join(
                self.data_dir, 'raw/images_test', img_name))
            new_img, dropped = self.process_img(img)
            if dropped:
                test_outliers_img.append(new_img)
                test_outliers_img_names.append(img_name)
            else:
                test_imgs.append(new_img.flatten())
                test_img_names.append(img_name)

        new_test_imgs = rbs.transform(np.array(test_imgs))

        os.mkdir(os.path.join(self.processed_dir, 'images_test'))
        for img_name, new_img in zip(test_img_names, new_test_imgs):
            torch.save(torch.Tensor(new_img.reshape(36, 36)), os.path.join(
                self.processed_dir, 'images_test/')+img_name)
        for img_name, new_img in zip(test_outliers_img_names, test_outliers_img):
            torch.save(torch.Tensor(new_img), os.path.join(
                self.processed_dir, 'images_test/')+img_name)

    def process_csv(self, taken_img_names):

        df_masks = self.masks.copy()

        patches_to_remove = ['well_0_patch_%s' % i for i in range(
            0, 165+1)] + ['well_1_patch_%s' % i for i in range(0, 450)]
        df_masks = df_masks.drop(patches_to_remove)
        new_indexs = []
        for img_name in df_masks.index:
            if 'well_1_patch_' in img_name:
                number = img_name.replace('well_1_patch_', '')
                new_number = int(number) - 450
                new_indexs.append('well_1_patch_%s' % new_number)
            else:
                new_indexs.append(img_name)
        print(new_indexs)
        df_masks.index = new_indexs
        df_masks_filtered = df_masks.loc[df_masks.index.isin(taken_img_names)]
        df_masks_filtered.to_csv(
            self.processed_dir + 'y_train.csv', index=True)
        self.masks = df_masks_filtered

    def process_img(self, img):
        if np.min(img) < -100:
            return img, True
        else:
            cleaned_img = img.copy()
            cleaned_img = np.nan_to_num(cleaned_img)  # Replace NaN with 0
            return cleaned_img, False

    def __len__(self):

        if not self.test:
            return len(self.masks)
        else:
            file_names = os.listdir(os.path.join(self.raw_dir, 'images_test'))
            return len(file_names)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        if not self.test:
            img_name = self.masks.index[idx]
            img_path = os.path.join(
                self.processed_dir, 'images_train', img_name+'.npy')
            image = torch.load(img_path).numpy()

            well = int(img_name[5:7].replace('_', ''))

            mask = self.masks.iloc[idx].values

        else:
            file_names = os.listdir(os.path.join(
                self.processed_dir, 'images_test'))
            img_path = os.path.join(
                self.processed_dir, 'images_test', file_names[idx])
            well = int(file_names[idx][5:7].replace('_', ''))
            image = torch.load(img_path).numpy()

            # Return a dummy mask for test data
            # Adjust the size according to your needs
            dummy_mask = np.zeros(1296)
            mask = dummy_mask

        # Convert single-channel to 3-channel RGB
        image = image[np.newaxis, :]
        image = np.tile(image, (3, 1, 1))
        mask = np.tile(mask, (3, 1, 1))
        mask = mask.reshape(3, 36, 36)

        random_states = [np.random.randint(0, 100), np.random.randint(0, 100)]

        if self.transform_img is None:
            image_tensor = torch.Tensor(image)
            mask_tensor = torch.Tensor(mask)
        elif isinstance(self.transform_img, RollTransform):
            image_tensor = self.transform_img(
                torch.Tensor(image), random_states)
            mask_tensor = self.transform_img(torch.Tensor(mask), random_states)
        else:
            image_tensor = self.transform_img(torch.Tensor(image))
            mask_tensor = self.transform_img(torch.Tensor(mask))

        return image_tensor, mask_tensor, torch.Tensor([well])


class CorroSeg():
    def __init__(self, data_dir, shuffle=True,
                 batch_size=64, valid_ratio=0.1, transform_img=None,
                 transform_test=None, test_params={'batch_size': 64, 'shuffle': False}):
        """Dataloader for the Corrosion Segmentation Challenge.

        Inputs:
            data_dir (string): Directory with all the images (train or test).
            shuffle (boolean): True if the data should be shuffled, False otherwise.
            batch_size (int): Size of the batch.
            valid_ratio (float): Ratio for the number of samples in the validation set.
            transform (callable, optional): Optional transform to be applied
                on training samples.
        """

        if transform_img is None:
            self.corroseg_dataset = CorroSegDataset(data_dir, test=False)
        else:
            self.corroseg_dataset = CorroSegDataset(
                data_dir, test=False, transform_img=transform_img[0])
            for i, trans_image in enumerate(transform_img):
                if i > 0:
                    self.corroseg_dataset = torch.utils.data.ConcatDataset([self.corroseg_dataset,
                                                                            CorroSegDataset(data_dir, test=False, transform_img=trans_image)])

        self.train_data, self.valid_data = torch.utils.data.random_split(
            self.corroseg_dataset, [1-valid_ratio, valid_ratio])
        self.train_dataloader = DataLoader(
            self.train_data, batch_size=batch_size, shuffle=shuffle)
        self.valid_dataloader = DataLoader(
            self.valid_data, batch_size=batch_size, shuffle=shuffle)

        self.test_data = CorroSegDataset(
            data_dir, transform_img=transform_test)
        self.test_dataloader = DataLoader(self.test_data, **test_params)

    def get_loaders(self):
        return self.train_dataloader, self.valid_dataloader, self.test_dataloader

    def get_datasets(self):
        return self.train_data, self.valid_data, self.test_data
