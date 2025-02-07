#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch
import torch.nn as nn
import model_embeddings

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
        super(CharDecoder,self).__init__()
        self.charDecoder=nn.LSTM(input_size=char_embedding_size,hidden_size=hidden_size)
        self.char_output_projection=nn.Linear(hidden_size,hidden_size,bias=True)
        self.MoreNonLinear=nn.Linear(hidden_size,len(target_vocab.char2id),bias=True)
        self.decoderCharEmb=torch.nn.Embedding(len(target_vocab.char2id),char_embedding_size,target_vocab.char2id['<pad>'])
        self.target_vocab=target_vocab
        ### END YOUR CODE



    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s_t in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.
        input_size=tuple(input.size())
        word_embed=self.decoderCharEmb(input)
        out,dec_hidden=self.charDecoder(word_embed,dec_hidden)
        s_t=self.char_output_projection(out)
        s_t=self.MoreNonLinear(torch.relu(s_t))
        assert(tuple(s_t.size())==(input_size[0],input_size[1],len(self.target_vocab.char2id)) )        
        return s_t,dec_hidden
        ### END YOUR CODEs


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).
#        print(char_sequence)
#        print(char_sequence)
        s_t,dec_hidden=self.forward(char_sequence[:-1],dec_hidden)
        loss=torch.nn.CrossEntropyLoss(reduction='sum',ignore_index=self.target_vocab.char2id['<pad>'])
        size=s_t.size()
        s_t=s_t.transpose(0,1).contiguous()
        char_sequence=char_sequence.transpose(0,1).contiguous()
        loss_sum=0
        for i in range(0,len(s_t)):
          loss_sum+=loss(s_t[i],char_sequence[i][1:])        
        return loss_sum
        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        batch_size=initialStates[0].size()[1]
        current_input=self.decoderCharEmb(torch.tensor([self.target_vocab.start_of_word]*batch_size,device=device).reshape(1,batch_size))
        
        s_t=None
        totalWord=torch.empty(1,batch_size,0,dtype=torch.long,device=device)
        for i in range(0,max_length):
            output,initialStates=self.charDecoder(current_input,initialStates)
            s_t=self.char_output_projection(output)
            s_t=self.MoreNonLinear(torch.relu(s_t))
            current_input=torch.argmax(s_t,dim=-1)
            totalWord=torch.cat((totalWord,current_input.view(1,batch_size,1)),-1)
            current_input=self.decoderCharEmb(current_input)
#            print(totalWord.size())
        totalWord=totalWord.squeeze(0)
        decodedWords=['']*batch_size
        for i in range(0,batch_size):
            for id in totalWord[i]:
                id=id.item()
                if(self.target_vocab.id2char[id]=='}'): break
                decodedWords[i]+=self.target_vocab.id2char[id]
        
        return decodedWords
        ### END YOUR CODE

