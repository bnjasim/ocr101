#!/usr/bin/env python
# coding: utf-8

import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import pickle
import random
import editdistance

# STEP1: TEXT DETECTION USING OPENCV

# Read the pickled character list
with open("all_chars_list.bin", "rb") as fp:   # Unpickling
    all_letters = pickle.load(fp)

n_letters = len(all_letters) + 1 # including blank; blank at the last index

def line_to_tensor(line):
    ifill = all_letters.index('.')
    linet = torch.tensor([all_letters.index(x) if x in all_letters else ifill for x in line])
    linet = linet.type(torch.FloatTensor)
    return linet

# Read all images and prepare train, test data
# Encapsulate an image patch and 
# its corresponding label as text
class Data:
    def __init__(self, img, text):
        self.img = img
        self.text = text
        self.target = line_to_tensor(text)
        self.target_length = torch.full(size=(1,), fill_value=len(text)) 
        self.output_length = torch.full(size=(1,), fill_value=img.shape[1])

        

# An image patch with its y location
# Used for sorting patches according to location
class Patch:
    def __init__(self, box, y):
        self.box = box
        self.ymin = y

def min_x(points):
    _x = [x[0][0] for x in points]
    return min(_x)

def max_x(points):
    _x = [x[0][0] for x in points]
    return max(_x)

def min_y(points):
    _y = [x[0][1] for x in points]
    return min(_y)

def max_y(points):
    _y = [x[0][1] for x in points]
    return max(_y)

def within_height(point_group, cnt):
    min1 = min_y(cnt)
    min2 = min_y(point_group)
    max1 = max_y(cnt)
    max2 = max_y(point_group)
    return (min1 >= min2 and min1 <= max2) or (max1 <= max2 and max1 >= min2)

# Read all the images in the folder data/
# Also read the (text) source files associated with the images
# match both line by line
all_data = []

for index in range(1, 149):
    img_path = os.path.join('data', str(index)+'.png')
    img = cv.imread(img_path)
    # Convert to Grayscale image
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Otsu's thresholding
    # th4, img_otsu = cv.threshold(img_gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    # print(th4)
    
    # Binary Thresholding and convert to binary image
    threshold = 75
    thresh, img_binary = cv.threshold(img_gray, threshold, 255, cv.THRESH_BINARY) 
    
    contours,_ = cv.findContours(img_binary, cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

    # We use a two pass custom made algorithm to find the bounding boxes
    # First pass of finding bounding boxes
    lines = []
    point_group = contours[0]

    for cnt in contours:
        if within_height(point_group, cnt) or within_height(cnt, point_group):
            point_group = np.append(point_group, cnt, axis=0)
        else:
            ch = cv.convexHull(point_group)
            lines.append(ch)
            point_group = cnt

    ch = cv.convexHull(point_group)
    lines.append(ch)

    # Second pass of finding bounding boxes
    img_lines = []
    for i in range(len(lines)-1):
        flag = True
        for j in range(i+1, len(lines)):
            if within_height(lines[i], lines[j]) or within_height(lines[j], lines[i]):
                lines[j] = np.append(lines[i], lines[j], axis=0)
                flag = False
                break

        if flag:
            x,y,w,h = cv.boundingRect(lines[i])
            # should be atleast 5x3 pixels and height should be less than 20px
            if h>2 and w>2 and h < 20 and w/h < 100:
                img_lines.append(lines[i])

    x,y,w,h = cv.boundingRect(lines[j])
    # should be atleast 5x3 pixels and height should be less than 20px
    if h>2 and w>2 and h < 20 and w/h < 100:
        img_lines.append(lines[j])

    # finally draw the bounding boxes
    # new_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_patches = []
    for line in img_lines:
        x,y,w,h = cv.boundingRect(line)
        # r = cv.rectangle(new_img,(x,y),(x+w,y+h),(0,255,0),1)
        patch = Patch(img_gray[y:y+h, x:x+w], y)
        img_patches.append(patch)

    # plt.imshow(new_img, 'gray'), plt.title('Bounding Boxes')
    # plt.show()
    # cv.imwrite('data/28_bounding_box.png', new_img) 

    # sort the image patches according to ymin
    img_patches = sorted(img_patches, key=lambda x: x.ymin)

    
    # Read the associated text file 
    text_path = os.path.join('data', str(index)+'.txt')
    with open(text_path) as f:
        text = f.read()

    text_lines = []
    for line in text.split('\n'):
        if len(line.strip())>0:
            text_lines.append(line.strip().replace('\t', ' '))

    # Check whether text lines and number of image patches match
    # Otherwise drop the entire frame from training
    if len(img_patches) != len(text_lines):
        print('Mismatch at image ' + str(index))
        continue
    
    # else
    # Add to data
    all_data += [Data(img.box, text) for img,text in zip(img_patches, text_lines)]



# Split to train data and test data
train_data = all_data[:2950] + all_data[3000:4500]
val_data = all_data[4500:5000]
test_data = all_data[5000:]



# STEP 2: TEXT RECOGNITION
# Train a GRU Model
n_pixels = 20 # width of a bounding box
n_hidden = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()       
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, 1, bidirectional=True) 
        self.linear = nn.Linear(2*hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input, h0):
        # input: seq_len x 1 x n_pixels (14)
        output, hn = self.gru(input, h0) # output: seq_len x 1 x 2*hidden_size
        # ToDo: Dropout
        output = self.linear(output) # output: seq_len x 1 x n_letters (96)
        output = self.softmax(output)
        return output

    def initHidden(self):
        return torch.zeros(2, 1, self.hidden_size).to(device)

rnn = RNN(n_pixels, n_hidden, n_letters).to(device)
h0 = rnn.initHidden()

optimizer = optim.Adam(rnn.parameters())

# ctc loss function
ctc_loss = nn.CTCLoss(blank=n_letters-1)

# find the predicted characters
def greedy_decode(output):
    output = output.squeeze(1)
    output_pos = torch.argmax(output, dim=1)
    
    blank = n_letters-1
    pred = []
    prev = blank
    for p in output_pos:
        if prev == blank and p != blank:
            pred.append(p.item())
        prev = p
       
    pred_s = [all_letters[i] for i in pred]
    return ''.join(pred_s)


# Training
n_epochs = 100
ckpt_dir = 'checkpoints/'
for epoch in range(n_epochs):
    print("Epoch: " + str(epoch+1))
    # Iterate over all the training data 
    for step in range(len(train_data)):
        t_data = train_data[step]
        patch = t_data.img
        h, w = np.shape(patch)
        # data augmentation: place the patch in different y locations
        r = random.randint(0, n_pixels-h)
        new_patch = np.zeros((n_pixels, w), 'uint8')
        new_patch[r:h+r, 0:] = patch
        
        optimizer.zero_grad()
        
        input = torch.tensor(np.transpose(new_patch)).unsqueeze(1)
        input = input.type(torch.FloatTensor).to(device)
        output = rnn(input, h0)
        # Compute CTC loss
        loss = ctc_loss(output, t_data.target, t_data.output_length, t_data.target_length)
        loss.backward()
        optimizer.step()
        
        if step % 500 == 0:
            print("Step: %d Loss: %.2f" % (step, loss.item()))
            
    ckpt_path = os.path.join(ckpt_dir, str(epoch+1)+'.pt')
    torch.save(rnn.state_dict(), ckpt_path)
    print('Checkpoint %d saved!' % (epoch+1))
    
    # Validation
    val_loss = 0
    for t_data in val_data:
        with torch.no_grad():
            patch = t_data.img
            h, w = np.shape(patch)
            # data augmentation: place the patch in different y locations
            r = random.randint(0, n_pixels-h)
            new_patch = np.zeros((n_pixels, w), 'uint8')
            new_patch[r:h+r, 0:] = patch
            input = torch.as_tensor(np.transpose(new_patch)).unsqueeze(1)
            input = input.type(torch.FloatTensor).to(device)
            output = rnn(input, h0)
            pred_line = greedy_decode(output)
            val_loss += editdistance.eval(pred_line, t_data.text) / len(t_data.text)
    
    print("Validation Loss = %.2f" % (100 * val_loss/len(val_data)))



# Inference
# Load saved Checkpoint
# rnn = RNN(n_pixels, n_hidden, n_letters)
# rnn.load_state_dict(torch.load(os.path.join(ckpt_dir, '20.pt')))

