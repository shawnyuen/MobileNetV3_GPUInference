#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 00:31:36 2019

@author: shaofengyuan
"""

import os
import torch
import argparse
from PIL import Image
from torchvision import transforms
from mobilenetv3 import MobileNetV3_Small
from mobilenetv3 import MobileNetV3_Large

def test(gpu_ids, which_model):
    """
    """
    with open('synset_words.txt', 'r') as f:
        classes = f.readlines()
    
    images_dir = os.getcwd()
    img_list = list()
    img_path = os.path.join(images_dir, '000_tench.jpg')
    img_list.append(img_path)
    img_path = os.path.join(images_dir, '001_goldfish.jpg')
    img_list.append(img_path)
    img_path = os.path.join(images_dir, '011_goldfinch.jpg')
    img_list.append(img_path)
    
    trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
    if gpu_ids != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
        
    if which_model == 'small':
        net = MobileNetV3_Small()
        state_dict = torch.load('mbv3_small.pth.tar')
        state_dict1 = state_dict['state_dict']
        net = torch.nn.DataParallel(net)
        net.load_state_dict(state_dict1)
        
        net.eval()
        
        print('[--Model--] using :{} model'.format(which_model))
        
        for item in img_list:
            img_pil = Image.open(item)
            print('[--Info--] path of image: {}'.format(item))
            print('[--Info--] size of original image: {}'.format(img_pil.size))
            
            img_torch = trans(img_pil)
            print('[--Info--] size of transformed image: {}'.format(img_torch.shape))
            
            img_torch = torch.unsqueeze(img_torch, 0).to(torch.device("cpu"))
            print('[--Info--] expand one dimension (batch size): {}'.format(img_torch.shape))
            output = net(img_torch)
            prob = torch.softmax(output, dim=1)
            print('[--Prob--] prob: {}'.format(prob.cpu().detach().numpy().max()))
            print('[--Class--] class: {}'.format(prob.cpu().detach().numpy().argmax()))
            print('[--ImageNet--] annotation: {}'.format(classes[prob.cpu().detach().numpy().argmax()]))
            
    elif which_model == 'large':
        net = MobileNetV3_Large()
        state_dict = torch.load('mbv3_large.pth.tar')
        state_dict1 = state_dict['state_dict']
        net = torch.nn.DataParallel(net)
        net.load_state_dict(state_dict1)
        
        net.eval()
        
        print('[--Model--] using :{} model'.format(which_model))
        
        for item in img_list:
            img_pil = Image.open(item)
            print('[--Info--] path of image: {}'.format(item))
            print('[--Info--] size of original image: {}'.format(img_pil.size))
            
            img_torch = trans(img_pil)
            print('[--Info--] size of transformed image: {}'.format(img_torch.shape))
            
            img_torch = torch.unsqueeze(img_torch, 0).to(torch.device("cpu"))
            print('[--Info--] expand one dimension (batch size): {}'.format(img_torch.shape))
            output = net(img_torch)
            prob = torch.softmax(output, dim=1)
            print('[--Prob--] prob: {}'.format(prob.cpu().detach().numpy().max()))
            print('[--Class--] class: {}'.format(prob.cpu().detach().numpy().argmax()))
            print('[--ImageNet--] annotation: {}'.format(classes[prob.cpu().detach().numpy().argmax()]))

            
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Pytorch 0.4.1, MobileNetV3 ImageNet Testing')
    
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0.')
    parser.add_argument('--which_model', type=str, default='small', help='chooses which model to use. [small | large]')
    
    args = parser.parse_args()
    
    test(args.gpu_ids, args.which_model)
