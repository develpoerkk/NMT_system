#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn
import torch
# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway


# End "do not change"

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings,self).__init__()

        ## A4 code
        pad_token_idx = vocab['<pad>']
        echar = 50
        self.embeddings = nn.Embedding(len(vocab.id2char), echar, padding_idx=pad_token_idx)
        ## End A4 code
        self.log_file=open('log.txt','w+')

        ### YOUR CODE HERE for part 1j

        self.embed_size=embed_size
        self.highWay=Highway(embed_size,embed_size)
        self.cnn=CNN(echar,embed_size)
        self.dropOut=nn.Dropout(p=0.3)
        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code
        self.log_file.write(str(self.embeddings.weight))

        ### YOUR CODE HERE for part 1j
        x_padded=self.embeddings(input)
        size=x_padded.size()
        x_reshape=x_padded.view(size[0]*size[1],size[3],size[2])
#        print(x_reshape.size())
        x_conv_out=self.cnn(x_reshape)
#        print(x_conv_out.size())
        x_conv_out=x_conv_out.view(size[0],size[1],self.embed_size) # 
#        print(x_conv_out.size())
        x_highway=self.highWay(x_conv_out)
        x_word_emb=self.dropOut(x_highway)
        return x_word_emb
        ### END YOUR CODE

