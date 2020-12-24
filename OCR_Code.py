#!/usr/bin/env python
# coding: utf-8

# In[664]:


import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch import optim

import os
import pickle
import random
import editdistance


# In[456]:


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


# In[464]:


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


# In[671]:


# Split to train data and test data
train_data = all_data[:2950] + all_data[3000:4500]
val_data = all_data[4500:5000]
test_data = all_data[5000:]


# In[495]:


plt.imshow(train_data[0].img, 'gray'), plt.show()


# In[673]:


# STEP 2: TEXT RECOGNITION
# Train a GRU Model
n_pixels = 20 # width of a bounding box
n_hidden = 128

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
        return torch.zeros(2, 1, self.hidden_size)

rnn = RNN(n_pixels, n_hidden, n_letters)
h0 = rnn.initHidden()

optimizer = optim.Adam(rnn.parameters())

# ctc loss function
ctc_loss = nn.CTCLoss(blank=n_letters-1)


# In[ ]:


# Training
n_epochs = 10
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
        input = input.type(torch.FloatTensor)
        output = rnn(input, h0)
        # Compute CTC loss
        loss = ctc_loss(output, t_data.target, t_data.output_length, t_data.target_length)
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
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
            input = torch.tensor(np.transpose(new_patch)).unsqueeze(1)
            input = input.type(torch.FloatTensor)
            output = rnn(input, h0)
            pred_line = greedy_decode(output)
            val_loss += editdistance.eval(pred_line, t_data.text) / len(t_data.text)
    
    print("Validation Loss = %.2f" % (100 * val_loss/len(val_data)))


# In[586]:


plt.imshow(test_data[2].img, 'gray'), plt.show()


# In[646]:


# Inference
# Load saved Checkpoint
rnn = RNN(n_pixels, n_hidden, n_letters)
rnn.load_state_dict(torch.load(os.path.join(ckpt_dir, '20.pt')))


# In[604]:


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


# In[669]:


index = 17
t_data = test_data[index]
with torch.no_grad():
    patch = t_data.img
    h, w = np.shape(patch)
    # data augmentation: place the patch in different y locations
    r = random.randint(0, n_pixels-h)
    new_patch = np.zeros((n_pixels, w), 'uint8')
    new_patch[r:h+r, 0:] = patch

    input = torch.tensor(np.transpose(new_patch)).unsqueeze(1)
    input = input.type(torch.FloatTensor)
    output = rnn(input, h0)
    pred_line = greedy_decode(output)
    print(pred_line)
    print(t_data.text)
    
plt.imshow(test_data[index].img, 'gray'), plt.show()

editdistance.eval(pred_line, t_data.text)


# In[628]:


pred, lsm = beam_decode(output, beam_size=10, blank=n_letters-1)
''.join([all_letters[i] for i in pred])


# In[616]:


import math
import collections

NEG_INF = -float("inf")

def make_new_beam():
    fn = lambda : (NEG_INF, NEG_INF)
    return collections.defaultdict(fn)

def logsumexp(*args):
    """
    Stable log sum exp.
    """
    if all(a == NEG_INF for a in args):
        return NEG_INF
    a_max = max(args)
    lsp = math.log(sum(math.exp(a - a_max)
                      for a in args))
    return a_max + lsp

def beam_decode(probs, beam_size=100, blank=0):
    """
    Performs inference for the given output probabilities.
    Arguments:
      probs: The output log probabilities (e.g. post-logsoftmax) for each
        time step. Should be an array of shape (time x 1 x output dim).
      beam_size (int): Size of the beam to use during inference.
      blank (int): Index of the CTC blank label.
    Returns the output label sequence and the corresponding negative
    log-likelihood estimated by the decoder.
    """
    probs = probs.squeeze(1)
    T, S = probs.shape

    # Elements in the beam are (prefix, (p_blank, p_no_blank))
    # Initialize the beam with the empty sequence, a probability of
    # 1 for ending in blank and zero for ending in non-blank
    # (in log space).
    beam = [(tuple(), (0.0, NEG_INF))]

    for t in range(T): # Loop over time

        # A default dictionary to store the next step candidates.
        next_beam = make_new_beam()

        for s in range(S): # Loop over vocab
            p = probs[t, s]

            # The variables p_b and p_nb are respectively the
            # probabilities for the prefix given that it ends in a
            # blank and does not end in a blank at this time step.
            for prefix, (p_b, p_nb) in beam: # Loop over beam

                # If we propose a blank the prefix doesn't change.
                # Only the probability of ending in blank gets updated.
                if s == blank:
                    n_p_b, n_p_nb = next_beam[prefix]
                    n_p_b = logsumexp(n_p_b, p_b + p, p_nb + p)
                    next_beam[prefix] = (n_p_b, n_p_nb)
                    continue

                # Extend the prefix by the new character s and add it to
                # the beam. Only the probability of not ending in blank
                # gets updated.
                end_t = prefix[-1] if prefix else None
                n_prefix = prefix + (s,)
                n_p_b, n_p_nb = next_beam[n_prefix]
                if s != end_t:
                    n_p_nb = logsumexp(n_p_nb, p_b + p, p_nb + p)
                else:
                  # We don't include the previous probability of not ending
                  # in blank (p_nb) if s is repeated at the end. The CTC
                  # algorithm merges characters not separated by a blank.
                  n_p_nb = logsumexp(n_p_nb, p_b + p)

                # *NB* this would be a good place to include an LM score.
                next_beam[n_prefix] = (n_p_b, n_p_nb)

                # If s is repeated at the end we also update the unchanged
                # prefix. This is the merging case.
                if s == end_t:
                    n_p_b, n_p_nb = next_beam[prefix]
                    n_p_nb = logsumexp(n_p_nb, p_nb + p)
                    next_beam[prefix] = (n_p_b, n_p_nb)

        # Sort and trim the beam before moving on to the
        # next time-step.
        beam = sorted(next_beam.items(),
                key=lambda x : logsumexp(*x[1]),
                reverse=True)
        beam = beam[:beam_size]

    best = beam[0]
    return best[0], -logsumexp(*best[1])


# In[ ]:




