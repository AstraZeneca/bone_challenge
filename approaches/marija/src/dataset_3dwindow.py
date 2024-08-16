import numpy as np
import os
from torch.utils.data import IterableDataset, Dataset
import torchio as tio

#ad-hock crop images in z to 400 (rest 200 slices are similar looking cortical bone)
MAX_IMG_LEN = 400

class TrainDataset(Dataset):
    def __init__(self, annotations_df, img_dir, seq_len=64, augment=True):
        self.annot = annotations_df
        self.img_dir = img_dir
        self.seq_len =seq_len
        self.tail = True
        self.augment = augment
        self.img_dir = img_dir
        self.img_len = 642 #hardcoded
        get_foreground = tio.ZNormalization.mean
        if augment:
            self.transform = tio.Compose([
                tio.Clamp(-1000, 4000),
                tio.Resample(2),
                tio.CropOrPad((224, 224, self.seq_len//2)),
                tio.ZNormalization(get_foreground),
                tio.RandomBlur(p=0.25),                    # blur 25% of times
                tio.RandomNoise(p=0.25),                   # Gaussian noise 25% of times   
                tio.RandomFlip((0, 1), p=1),
                tio.OneOf({                                # either
                    tio.RandomMotion(): 1,                 # random motion artifact
                    tio.RandomSpike(): 2,                  # or spikes
                    tio.RandomGhosting(): 2,               # or ghosts
                }, p=0.5)])
        else:
            self.transform = tio.Compose([
                    tio.Clamp(-1000, 4000),
                    tio.Resample(2),
                    tio.CropOrPad((224, 224, self.seq_len//2)),
                    tio.ZNormalization(get_foreground)])

    def get_range(self, image_len, gp_index, tail):    
        if tail:
            possible_center_pre = np.arange(1+self.seq_len/2, gp_index-5-self.seq_len/2)
            possible_center_post = np.arange(gp_index+5+self.seq_len/2,  MAX_IMG_LEN-self.seq_len/2-1)
            possible_center = np.concatenate([possible_center_pre, possible_center_post])
            center = np.random.choice(possible_center)
            start = int(round(center-self.seq_len/2))
            end = start+self.seq_len
            return start, end
        else:
            offset = int(np.round(np.random.uniform(5,self.seq_len-5)))
            start = min(image_len-self.seq_len, max(0, gp_index-offset+1))
            end = start + self.seq_len
            return start, end
    
    def __len__(self):
        return 2*len(self.annot)


    def __getitem__(self, idx_):
        idx = idx_//2
        tail = idx_%2
        img_bname = self.annot.iloc[idx]['Image Name']+'.nii'
        img_path = os.path.join(self.img_dir, img_bname)
        img = tio.ScalarImage(img_path)
        zindx = self.annot.iloc[idx]['Growth Plate Index']
        start, finish = self.get_range(self.img_len, zindx, tail)
        clf_pred = int((start<zindx) & (finish > zindx))
        reg_pred = (zindx-start)/self.seq_len if clf_pred else -1
        img = tio.Crop((0,0, 0, 0, int(start), int(self.img_len-finish)))(img)
        img = self.transform(img).data
        img = img.permute(0, 3, 1, 2)
        return {'image_name':img_bname, 'zindx':zindx, 'input_seq':img, 'clf':clf_pred, 'reg':reg_pred ,'start_indx':start, 'end_indx':finish}
            
    
class ValidDataset(IterableDataset):
    def __init__(self, annotations_df, img_dir, seq_len=64, stride=16):
        self.annot = annotations_df
        self.img_dir = img_dir
        self.seq_len = seq_len
        self.nimages = len(self.annot)
        self.stride = stride
        self.img_len = 642
        get_foreground = tio.ZNormalization.mean
        self.transform = tio.Compose([
                    tio.Clamp(-1000, 4000),
                    tio.Resample(2),
                    tio.CropOrPad((224, 224, self.seq_len//2)),
                    tio.ZNormalization(get_foreground)])

    def generate(self):
        img_indx = 0
        while img_indx<self.nimages:
            img_bname = self.annot.iloc[img_indx ]['Image Name']+'.nii'
            img_path = os.path.join(self.img_dir, img_bname)
            img = tio.ScalarImage(img_path)
            if 'Growth Plate Index' in self.annot.columns:
                zindx = self.annot.iloc[img_indx]['Growth Plate Index']
            else:
                zindx = -1
            start = 0
            finish = start + self.seq_len
            while finish < MAX_IMG_LEN:
                img_crop = tio.Crop((0,0, 0, 0, int(start), int(self.img_len-finish)))(img)
                img_crop = self.transform(img_crop).data
                img_crop = img_crop.permute(0, 3, 1, 2)
                clf_pred = int((start<=zindx) & (finish > zindx))
                reg_pred = (zindx-start)/self.seq_len if clf_pred else -1
                yield {'image_name':img_bname, 'zindx':zindx, 'input_seq':img_crop, 'clf':clf_pred, 'reg':reg_pred, 'start_indx':start, 'end_indx':finish}
                start = start + self.stride
                finish = start + self.seq_len
            img_indx +=1 

    def __iter__(self):
        return iter(self.generate())
