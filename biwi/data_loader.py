import os
import torch
from collections import defaultdict
from torch.utils import data
import copy
import numpy as np
import pickle
from tqdm import tqdm
import random,math
from transformers import Wav2Vec2FeatureExtractor,Wav2Vec2Processor
import librosa    

class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data,subjects_dict,data_type="train"):
        self.data = data
        self.len = len(self.data)
        self.subjects_dict = subjects_dict
        self.data_type = data_type
        self.one_hot_labels = np.eye(len(subjects_dict["train"]))

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        file_name = self.data[index]["name"]
        audio = self.data[index]["audio"]
        vertice = self.data[index]["vertice"]
        template = self.data[index]["template"]
        if self.data_type == "train":
            one_hot = self.one_hot_labels[self.subjects_dict["train"].index(file_name[:2])]
        else:
            one_hot = self.one_hot_labels
        return torch.FloatTensor(audio),torch.FloatTensor(vertice), torch.FloatTensor(template), torch.FloatTensor(one_hot), file_name

    def __len__(self):
        return self.len

def read_data(args,template_file = "BIWI_data/templates.pkl"):
    print("Loading data...")
    data = defaultdict(dict)
    train_data = []
    valid_data = []
    test_data = []

    audio_path = args.wav_path
    vertices_path = args.vertices_path
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin)
    
    for r, ds, fs in os.walk(audio_path):
        for f in tqdm(fs):
            if f.endswith("wav"):
                wav_path = os.path.join(r,f)
                speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
                input_values = np.squeeze(processor(speech_array,sampling_rate=16000).input_values)
                key = f.replace("wav", "npy")
                data[key]["audio"] = input_values

                temp = templates[f.split("_")[0]]
                data[key]["name"] = f
                data[key]["template"] = temp.reshape((-1)) # (23370, 3) -> (23370*3)
                vertice_path = os.path.join(vertices_path,f.replace("wav", "npy"))
                data[key]["vertice"] = np.load(vertice_path,allow_pickle=True)

    subjects_dict = {}
    subjects_dict["train"] = [i for i in args.train_subjects.split(" ")]
    subjects_dict["val"] = [i for i in args.val_subjects.split(" ")]
    subjects_dict["test"] = [i for i in args.test_subjects.split(" ")]
   
    for k, v in data.items():
        subject_id = k[:2]
        sentence_id = int(k[4:6])
        if subject_id in subjects_dict["train"] and sentence_id>0 and sentence_id<=32:
            train_data.append(v)
        if subject_id in subjects_dict["val"] and sentence_id>32 and sentence_id<=36:
            valid_data.append(v)
        if subject_id in subjects_dict["test"] and sentence_id>36 and sentence_id<=40:
            test_data.append(v)

    print(len(train_data), len(valid_data), len(test_data))
    return train_data, valid_data, test_data, subjects_dict

def get_dataloaders(args):
    dataset = {}
    train_data, valid_data, test_data, subjects_dict = read_data(args)
    train_data = Dataset(train_data,subjects_dict,"train")
    dataset["train"] = data.DataLoader(dataset=train_data, batch_size=1, shuffle=True)
    valid_data = Dataset(valid_data,subjects_dict,"val")
    dataset["valid"] = data.DataLoader(dataset=valid_data, batch_size=1, shuffle=False)
    test_data = Dataset(test_data,subjects_dict,"test")
    dataset["test"] = data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)
    return dataset

if __name__ == "__main__":
    get_dataloaders()
    