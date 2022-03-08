import re, random, math
import numpy as np
import argparse
from tqdm import tqdm
import os, shutil
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_loader import get_dataloaders
from faceformer import Faceformer

def trainer(args, train_loader, dev_loader, model, optimizer, criterion, epoch=100):
    save_path = os.path.join(args.data_path,args.save_path)
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    train_subjects_list = [i for i in args.train_subjects.split(" ")]
    iteration = 0
    for e in range(epoch+1):
        loss_log = []
        # train
        model.train()
        pbar = tqdm(enumerate(train_loader),total=len(train_loader))
        optimizer.zero_grad()

        for i, (audio, vertice, template, one_hot, file_name) in pbar:
            iteration += 1
            # to gpu
            audio, vertice, template, one_hot  = audio.to(device='cuda'), vertice.to(device='cuda'), template.to(device='cuda'), one_hot.to(device='cuda')
            loss = model(audio, template,  vertice, one_hot, criterion)
            loss.backward()
            loss_log.append(loss.item())
            if i % args.gradient_accumulation_steps==0:
                optimizer.step()
                optimizer.zero_grad()

            pbar.set_description("(Epoch {}, iteration {}) TRAIN LOSS:{:.7f}".format((e+1), iteration ,np.mean(loss_log)))
        # validation
        valid_loss_log = []
        model.eval()
        
        for audio, vertice, template, one_hot_all,file_name in dev_loader:
            # to gpu
            audio, vertice, template, one_hot_all= audio.to(device='cuda'), vertice.to(device='cuda'), template.to(device='cuda'), one_hot_all.to(device='cuda')
            if file_name[0][:2] in train_subjects_list:
                condition_subject = file_name[0][:2]
                iter = train_subjects_list.index(condition_subject)
                one_hot = one_hot_all[:,iter,:]
                loss = model(audio, template,  vertice, one_hot, criterion)
                valid_loss_log.append(loss.item())
            else:
                for iter in range(one_hot_all.shape[-1]):
                    condition_subject = train_subjects_list[iter]
                    one_hot = one_hot_all[:,iter,:]
                    loss = model(audio, template,  vertice, one_hot, criterion)
                    valid_loss_log.append(loss.item())
                        
        current_loss = np.mean(valid_loss_log)
        if (e > 0 and e % 25 == 0) or e == args.max_epoch:
            torch.save(model.state_dict(), os.path.join(save_path,'{}_model.pth'.format(e)))
        print("epcoh: {}, current loss:{:.7f}".format(e+1,current_loss)) 

    return model

@torch.no_grad()
def test(args, model, test_loader,epoch):
    result_path = os.path.join(args.data_path,args.result_path)
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.makedirs(result_path)
    save_path = os.path.join(args.data_path,args.save_path)
    train_subjects_list = [i for i in args.train_subjects.split(" ")]

    model.load_state_dict(torch.load(os.path.join(save_path, '{}_model.pth'.format(epoch))))
    model = model.to(torch.device('cuda'))
    model.eval()
    for audio, vertice, template, one_hot_all, file_name in test_loader:
        # to gpu
        audio, vertice, template, one_hot_all= audio.to(device='cuda'), vertice.to(device='cuda'), template.to(device='cuda'), one_hot_all.to(device='cuda')
        if file_name[0].split("_sentence")[0] in train_subjects_list:
            condition_subject = file_name[0].split("_sentence")[0]
            iter = train_subjects_list.index(file_name[0].split("_sentence")[0])
            one_hot = one_hot_all[:,iter,:]
            prediction = model.predict(audio, template, one_hot)
            prediction = prediction.squeeze() # (seq_len, V*3)
            np.save(os.path.join(result_path, file_name[0].split(".")[0]+"_condition_"+condition_subject+".npy"), prediction.detach().cpu().numpy())
        else:
            for iter in range(one_hot_all.shape[-1]):
                condition_subject = train_subjects_list[iter]
                one_hot = one_hot_all[:,iter,:]
                prediction = model.predict(audio, template, one_hot)
                prediction = prediction.squeeze() # (seq_len, V*3)
                np.save(os.path.join(result_path, file_name[0].split(".")[0]+"_condition_"+condition_subject+".npy"), prediction.detach().cpu().numpy())
         
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--vertice_dim", type=int, default=5023*3)
    parser.add_argument("--wav_path", type=str, default= "VOCASET/wav")
    parser.add_argument("--vertices_path", type=str, default="VOCASET/vertices_npy")#
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    #follow the same split as VOCA: https://github.com/TimoBolkart/voca/blob/master/config_parser.py
    parser.add_argument("--train_subjects", type=str, default="FaceTalk_170728_03272_TA"
       " FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA"
       " FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA"
       " FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA")
    parser.add_argument("--val_subjects", type=str, default="FaceTalk_170811_03275_TA"
       " FaceTalk_170908_03277_TA")
    parser.add_argument("--test_subjects", type=str, default="FaceTalk_170809_00138_TA"
       " FaceTalk_170731_00024_TA")
    parser.add_argument("--data_path", type=str, default="VOCASET")
    parser.add_argument("--save_path", type=str, default="save")
    parser.add_argument("--result_path", type=str, default="result")
    args = parser.parse_args()

    model = Faceformer(args)
    print("model parameters: ", count_parameters(model))
    # to cuda
    assert torch.cuda.is_available()
    model = model.to(torch.device('cuda'))
    # load data
    dataset = get_dataloaders(args)
    # loss
    criterion = nn.MSELoss()
    # Train the model
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=args.lr)
    model = trainer(args, dataset["train"], dataset["valid"],model, optimizer, criterion, epoch=args.max_epoch)
    # Test the model
    test(args, model, dataset["test"], epoch=args.max_epoch)
    
if __name__=="__main__":
    main()