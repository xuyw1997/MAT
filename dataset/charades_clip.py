import torch
from torch.utils.data import Dataset
import json
import torchtext
import torch.nn as nn
import numpy as np
import os
import csv
import torch.nn.functional as F
from PIL import Image
from .build import DATASET_REGISTRY
import dataset.utils as utils



@DATASET_REGISTRY.register()
class CharadesClip(Dataset):

    vocab = torchtext.vocab.pretrained_aliases["glove.6B.300d"]()
    vocab.itos.extend(['<unk>'])
    vocab.stoi['<unk>'] = vocab.vectors.shape[0]
    vocab.vectors = torch.cat([vocab.vectors, torch.zeros(1, vocab.dim)], dim=0)
    word_embedding = nn.Embedding.from_pretrained(vocab.vectors)
    print("word vector loaded.")
    def __init__(self, cfg, split):
        super(CharadesClip, self).__init__()


        self.cfg = cfg
        self.split = split
        self.num_max_word = cfg.DATA.NUM_MAX_WORD
        self.data_path = cfg.DATA.DATA_PATH
        self.num_sample_frame = cfg.DATA.NUM_SAMPLE_FRAME
        self.annotations = None
        self.drop_last = split == 'train'
        self.num_stream = cfg.TRAIN.BATCH_SIZE if split == 'train' else cfg.TEST.BATCH_SIZE
        self.num_clip_per_video = cfg.DATA.NUM_SAMPLE_FRAME // cfg.DATA.WINDOW_SIZE
        self.window_size = cfg.DATA.WINDOW_SIZE
        self.input_size = cfg.DATA.INPUT_SIZE
        self.mlm_p = cfg.DATA.MLM_P
        
        with open(os.path.join(cfg.DATA.NUM_FRAME_PATH),'r') as f:
           self.num_video_frame = json.load(f)
           
        self.durations = {}
        with open(os.path.join(cfg.DATA.ANNO_PATH, 'Charades_v1_{}.csv'.format(self.split))) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.durations[row['id']] = float(row['length'])

        with open(cfg.DATA.VOCAB_PATH, 'r') as f:
            tmp = json.load(f)
            self.itos = tmp['words']
        self.stoi = {}
        for i, w in enumerate(self.itos):
            self.stoi[w] = i
            
        anno_file = open(os.path.join(cfg.DATA.ANNO_PATH, "charades_sta_{}.txt".format(self.split)), 'r')
        annotations = []
        for line in anno_file:
            anno, sent = line.split("##")
            sent = sent.split('.\n')[0]
            vid, s_time, e_time = anno.split(" ")
            s_time = float(s_time)
            e_time = min(float(e_time), self.durations[vid])
            desc_split = sent.split()
            # desc_split, txt_mask = utils.trim_pad_to_fixed_length(desc_split, self.num_max_word)
            txt_mask = torch.ones(len(desc_split))
            if s_time < e_time:
                annotations.append({
                    'video': vid, 
                    'times': [s_time, e_time], 
                    'description': sent, 
                    'desc_split': desc_split, 
                    'txt_mask': txt_mask, 
                    'duration': self.durations[vid]
                    })
        anno_file.close()
        self.annotations = annotations
        self._make_index_mapping(list(range(len(self.annotations))))
        self.transform = utils.get_transform(cfg.DATA.INPUT_SIZE, cfg.DATA.MEAN, cfg.DATA.STD, split)

    def __len__(self):
        return sum([len(s) for s in self.index_mapping])

    def _make_index_mapping(self, perm):
        if self.drop_last:
            num_video_per_stream = len(self.annotations) // self.num_stream
        else:
            num_video_per_stream = (len(self.annotations) + self.num_stream - 1) // self.num_stream
    
        self.index_mapping = [[] for _ in range(self.num_stream)]
        
        cur_stream = 0
        num_videos = num_video_per_stream * self.num_stream if self.drop_last else len(self.annotations)
        for id in range(num_videos):
            anno = self.annotations[perm[id]]
            for i in range(self.num_clip_per_video):
                self.index_mapping[cur_stream].append({'video': anno['video'], 'clip': i, 'id_in_anno': perm[id]})
            cur_stream = (cur_stream + 1) % self.num_stream
        

        max_stream_num_clip = max([len(s) for s in self.index_mapping])
        for s in self.index_mapping:
            num_pad = max_stream_num_clip - len(s)
            for i in range(num_pad):
                s.append({'video': 'pad_video', 'clip': -1, 'id_in_anno': -1})
    
    def shuffle(self):
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator = torch.Generator()
        generator.manual_seed(seed)
        
        perm = torch.randperm(len(self.annotations), generator=generator).tolist()
        self._make_index_mapping(perm)
        print("dataset shuffled.")


    def __getitem__(self, index):
        stream_id = index % self.num_stream
        id_in_stream = index // self.num_stream

        video_id = self.index_mapping[stream_id][id_in_stream]['video']
        clip_id = self.index_mapping[stream_id][id_in_stream]['clip']


        anno = self.annotations[self.index_mapping[stream_id][id_in_stream]['id_in_anno']]
        desc_split = anno['desc_split']
        txt_mask = anno['txt_mask']
        duration = anno['duration']
        times = anno['times']
        times = torch.tensor(times, dtype=torch.float)
        
        word_label = [self.stoi.get(w.lower(), len(self.itos)-1) for w in desc_split]
        range_i = range(len(word_label))
        if self.mlm_p > 0:
            word_mask = [1. if np.random.uniform(0,1)< self.mlm_p else 0. for _ in range_i]
            if np.sum(word_mask) == 0.:
                mask_i = np.random.choice(range_i)
                word_mask[mask_i] = 1.
            if np.sum(word_mask) == len(word_mask):
                unmask_i = np.random.choice(range_i)
                word_mask[unmask_i] = 0.
        else:
            word_mask = [0. for _ in range_i]
                
        word_label = torch.tensor(word_label, dtype=torch.long)
        word_mask = torch.tensor(word_mask, dtype=torch.float)
        
        word_idxs = torch.tensor([self.vocab.stoi.get(w.lower(), 400000) for w in desc_split], dtype=torch.long)
        word_vectors = self.word_embedding(word_idxs)

        if video_id == 'pad_video':
            # frames = torch.zeros(3 * self.window_size, self.input_size, self.input_size)
            frames = torch.zeros(3 , self.window_size, self.input_size, self.input_size)
            clip_mask = False
        else:
            frame_seq = utils.get_frame_sequence(self.cfg, self.num_video_frame[video_id], rand_offset=self.split == 'train')
            frame_seq = frame_seq[clip_id * self.window_size: (clip_id + 1) * self.window_size]
            frames = self.get_video_frames(video_id, frame_seq)
            frames = self.transform(frames)
            clip_mask = True

        # Time scaled to same size
        # label = utils.label_time_scale(self.num_sample_frame, duration, times)
        label = times / duration 
        assert label[0] < label[1], f'annotation error, start: {label[0]} and end: {label[1]}'
        assert label[1] <= 1, f'annotation error, end: {label[1]} bigger than 1'
        
        return frames, word_vectors, txt_mask, word_label, word_mask, label, times, duration, clip_mask
        

    def get_video_frames(self, video_id, frame_seq):
        images = []
        for idx in frame_seq:
            images.append(Image.open(os.path.join(self.data_path, video_id, f'{video_id}-{idx+1:06}.jpg')).convert('RGB'))
    
        return images






