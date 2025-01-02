import os
import cv2
import sys
import pdb
import six
import glob
import time
import torch
import random
import pandas
import warnings
# from multicueextra import multicue_extra
import pickle
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
# import pyarrow as pa
from PIL import Image
import torch.utils.data as data
import matplotlib.pyplot as plt
from utils import video_augmentation
from torch.utils.data.sampler import Sampler

sys.path.append("..")


class BaseFeeder(data.Dataset):
    def __init__(self, prefix, gloss_dict, drop_ratio=1, num_gloss=-1, mode="train", transform_mode=True,
                 datatype="lmdb"):
        self.mode = mode
        self.ng = num_gloss
        self.prefix = prefix
        self.dict = gloss_dict
        self.data_type = datatype
        self.feat_prefix = f"{prefix}/features/fullFrame-256x256px/{mode}"
        self.transform_mode = "train" if transform_mode else "test"
        self.inputs_list = np.load(f"./preprocess/phoenix2014/{mode}_info.npy", allow_pickle=True).item()
        # self.inputs_list = np.load(f"{prefix}/annotations/manual/{mode}.corpus.npy", allow_pickle=True).item()
        # self.inputs_list = np.load(f"{prefix}/annotations/manual/{mode}.corpus.npy", allow_pickle=True).item()
        # self.inputs_list = dict([*filter(lambda x: isinstance(x[0], str) or x[0] < 10, self.inputs_list.items())])
        print(mode, len(self))
        self.data_aug = self.transform()
        print("")

    def __getitem__(self, idx):
        if self.data_type == "video":
            # multicue = multicue_extra()
            input_data, label, fi, Facebound, Rhandbound, Lhandbound = self.read_video(idx)
            input_data, label = self.normalize(input_data, label)
            # input_data, label = self.normalize(input_data, label, fi['fileid'])
            return input_data, torch.LongTensor(label), self.inputs_list[idx]['original_info'],Facebound, Rhandbound, Lhandbound
        elif self.data_type == "lmdb":
            input_data, label, fi = self.read_lmdb(idx)
            input_data, label = self.normalize(input_data, label)
            return input_data, torch.LongTensor(label), self.inputs_list[idx]['original_info']
        else:
            input_data, label = self.read_features(idx)
            return input_data, label, self.inputs_list[idx]['original_info']

    def read_video(self, index, num_glosses=-1):
        if self.mode == 'train':
            path = '/home/zy/zy/dataset/Data_cue/feature/train/'
            path2 = '/home/zy/zy/dataset/Data_cue/new_bounding/unindex/'
        elif self.mode == 'test':
            path = '/home/zy/zy/dataset/Data_cue/feature/test/'
            path2 = '/home/zy/zy/dataset/Data_cue/new_bounding/unindex_test/'
        else:
            path = '/home/zy/zy/dataset/Data_cue/feature/dev/'
            path2 = '/home/zy/zy/dataset/Data_cue/new_bounding/unindex_dev/'

    # def read_video(self, index, num_glosses=-1):
    #     if self.mode == 'train':
    #         path = '/hdd1/zy/dataset/Data_cue/keypoint/train/'
    #         path2 = '/hdd1/zy/dataset/Data_cue/key_unindex/train/'
    #     elif self.mode == 'test':
    #         path = '/hdd1/zy/dataset/Data_cue/keypoint/test/'
    #         path2 = '/hdd1/zy/dataset/Data_cue/key_unindex/test/'
    #     else:
    #         path = '/hdd1/zy/dataset/Data_cue/keypoint/dev/'
    #         path2 = '/hdd1/zy/dataset/Data_cue/key_unindex/dev/'

        fi = self.inputs_list[index]
        img_folder = os.path.join(self.prefix, "features/fullFrame-256x256px/" + fi['folder'])
        # img_folder = os.path.join(self.prefix, "features/fullFrame-256x256px/train/06April_2010_Tuesday_heute_default-10/1/*.png")

        img_list = sorted(glob.glob(img_folder))
        aa = fi['fileid']
        # aa = '06April_2010_Tuesday_heute_default-10'
        unindex_name = open(path2 +str(aa)+'-.pkl', "rb")
        result_unindex = pickle.load(unindex_name)
        for aaaa in result_unindex[0]:
            nnnn =self.prefix+"/features/fullFrame-256x256px"+aaaa[94:]

            # nnnn =self.prefix+"/features/fullFrame-256x256px"+aaaa[59:]
            img_list.remove(nnnn)
        bounding_name = open(path +str(aa)+'.pkl', "rb")
        result_bounding = pickle.load(bounding_name)

        # Face = torch.tensor(result_bounding[0])
        # Lhand = torch.tensor(result_bounding[1])
        # Rhand = torch.tensor(result_bounding[2])
        Face = torch.stack(result_bounding[0])
        Rhand = torch.stack(result_bounding[1])
        Lhand = torch.stack(result_bounding[2])
        label_list = []
        for phase in fi['label'].split(" "):
            if phase == '':
                continue
            if phase in self.dict.keys():
                label_list.append(self.dict[phase][0])

        return [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in img_list], label_list, fi, Face, Rhand, Lhand

    def read_features(self, index):
        # load file info
        fi = self.inputs_list[index]
        data = np.load(f"./features/{self.mode}/{fi['fileid']}_features.npy", allow_pickle=True).item()
        return data['features'], data['label']

    def normalize(self, video, label, file_id=None):
        video, label = self.data_aug(video, label, file_id)
        video = video.float() / 127.5 - 1
        return video, label

    def transform(self):
        if self.transform_mode == "train":
            print("Apply training transform.")
            return video_augmentation.Compose([
                # video_augmentation.CenterCrop(224),
                # video_augmentation.WERAugment('/lustre/wangtao/current_exp/exp/baseline/boundary.npy'),
                video_augmentation.RandomCrop(224),
                video_augmentation.RandomHorizontalFlip(0.5),
                video_augmentation.ToTensor(),
                # video_augmentation.TemporalRescale(0.2),
                # video_augmentation.Resize(1.0),
            ])
        else:
            print("Apply testing transform.")
            return video_augmentation.Compose([
                video_augmentation.CenterCrop(224),
                # video_augmentation.Resize(1.0),
                video_augmentation.ToTensor(),
            ])

    def byte_to_img(self, byteflow):
        unpacked = pa.deserialize(byteflow)
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        return img

    @staticmethod
    def collate_fn(batch):
        # batch = [item for item in sorted(batch, key=lambda x: len(x[0]), reverse=True)]
        # video, label, info, Face, Rhands, Lhands  = list(zip(*batch))
        # if len(video[0].shape) > 3:
        #     max_len = len(video[0])
        #     video_length = torch.LongTensor([np.ceil(len(vid) / 4.0) * 4 + 12 for vid in video])
        #     left_pad = 6
        #     right_pad = int(np.ceil(max_len / 4.0)) * 4 - max_len + 6
        #     max_len = max_len + left_pad + right_pad
        batch = [item for item in sorted(batch, key=lambda x: len(x[0]), reverse=True)]
        video, label, info, Face, Rhands, Lhands  = list(zip(*batch))
        if len(video[0].shape) > 3:
            max_len = len(video[0])
            video_length = torch.LongTensor([np.ceil(len(vid) / 4.0) * 4 + 12 for vid in video])
            left_pad = 6
            right_pad = int(np.ceil(max_len / 4.0)) * 4 - max_len + 6
            max_len = max_len + left_pad + right_pad
            padded_video = [torch.cat(
                (
                    vid[0][None].expand(left_pad, -1,-1,-1),
                    vid,
                    vid[-1][None].expand(max_len - len(vid) - left_pad, -1,-1,-1),
                )
                , dim=0)
                for vid in video]
            padded_video = torch.stack(padded_video)

            padded_Face = [torch.cat(
                (
                    Face1[0][None].expand(left_pad, -1,-1,-1),
                    Face1,
                    Face1[-1][None].expand(max_len - len(Face1) - left_pad, -1,-1,-1),
                )
                , dim=0)
                for Face1 in Face]
            padded_Face = torch.stack(padded_Face)
            padded_Rhands = [torch.cat(
                (
                    Rhands1[0][None].expand(left_pad, -1, -1, -1),
                    Rhands1,
                    Rhands1[-1][None].expand(max_len - len(Rhands1) - left_pad, -1, -1, -1),
                )
                , dim=0)
                for Rhands1 in Rhands]
            padded_Rhands = torch.stack(padded_Rhands)

            padded_Lhands = [torch.cat(
                (
                    Lhands1[0][None].expand(left_pad, -1, -1, -1),
                    Lhands1,
                    Lhands1[-1][None].expand(max_len - len(Lhands1) - left_pad, -1, -1, -1),
                )
                , dim=0)
                for Lhands1 in Lhands]
            padded_Lhands = torch.stack(padded_Lhands)
            # padded_Face = [torch.cat(
            #     (
            #         Face1[0][None].expand(left_pad, -1),
            #         Face1,
            #         Face1[-1][None].expand(max_len - len(Face1) - left_pad, -1),
            #     )
            #     , dim=0)
            #     for Face1 in Face]
            # padded_Face = torch.stack(padded_Face)

            # padded_Rhands = [torch.cat(
            #     (
            #         Rhands1[0][None].expand(left_pad, -1),
            #         Rhands1,
            #         Rhands1[-1][None].expand(max_len - len(Rhands1) - left_pad, -1),
            #     )
            #     , dim=0)
            #     for Rhands1 in Rhands]
            # padded_Rhands = torch.stack(padded_Rhands)
            #
            # padded_Lhands = [torch.cat(
            #     (
            #         Lhands1[0][None].expand(left_pad,-1),
            #         Lhands1,
            #         Lhands1[-1][None].expand(max_len - len(Lhands1) - left_pad,-1),
            #     )
            #     , dim=0)
            #     for Lhands1 in Lhands]
            # padded_Lhands = torch.stack(padded_Lhands)
        else:
            max_len = len(video[0])
            video_length = torch.LongTensor([len(vid) for vid in video])
            padded_video = [torch.cat(
                (
                    vid,
                    vid[-1][None].expand(max_len - len(vid), -1),
                )
                , dim=0)
                for vid in video]
            padded_video = torch.stack(padded_video).permute(0, 2, 1)
        label_length = torch.LongTensor([len(lab) for lab in label])
        if max(label_length) == 0:
            return padded_video, video_length, [], [], info
        else:
            padded_label = []
            for lab in label:
                padded_label.extend(lab)
            padded_label = torch.LongTensor(padded_label)
            return padded_video, video_length, padded_label, label_length, padded_Face, padded_Lhands,padded_Rhands,info

    def __len__(self):
        return len(self.inputs_list) - 1

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time


if __name__ == "__main__":
    feeder = BaseFeeder()
    dataloader = torch.utils.data.DataLoader(
        dataset=feeder,
        batch_size=1,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    for data in dataloader:
        pdb.set_trace()
