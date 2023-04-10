# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""

from __future__ import print_function
import dataclasses
import logging
import os
from selectors import EpollSelector
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple, List, Union, Any, NewType, Callable
from enum import Enum
from filelock import FileLock
import math
import json
from dataclasses import dataclass
from collections.abc import Mapping

import os
os.environ["WANDB_DISABLED"] = "true"

import numpy
import numpy as np

import spacy
import scispacy
nlp=spacy.load("en_core_sci_scibert")

from torch import nn
import torch
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss, Parameter
import torch.nn.functional as F
from torch.utils.data import Dataset
# from torch.utils.data.dataloader import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse


import transformers
from transformers import ElectraTokenizer, EvalPrediction, GlueDataset, ElectraModel, ElectraConfig, ElectraPreTrainedModel
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    #glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
)
from transformers import Trainer
from transformers.trainer_gpnn import Trainer_GPNN
from transformers.data.processors.utils import InputExample, InputFeatures, DataProcessor
from transformers.file_utils import ModelOutput, is_tf_available
from transformers.tokenization_utils import PreTrainedTokenizer

from metrics import compute_metrics

logger = logging.getLogger(__name__)

global max_length_edge

class Encoder(nn.Module):
    """
    Encoder class for Pointer-Net
    """

    def __init__(self, embedding_dim,
                 hidden_dim,
                 n_layers,
                 dropout,
                 bidir):
        """
        Initiate Encoder

        :param Tensor embedding_dim: Number of embbeding channels
        :param int hidden_dim: Number of hidden units for the LSTM
        :param int n_layers: Number of layers for LSTMs
        :param float dropout: Float between 0-1
        :param bool bidir: Bidirectional
        """

        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim//2 if bidir else hidden_dim
        self.n_layers = n_layers*2 if bidir else n_layers
        self.bidir = bidir
        self.lstm = nn.LSTM(embedding_dim,
                            self.hidden_dim,
                            n_layers,
                            dropout=dropout,
                            bidirectional=bidir)

        # Used for propagating .cuda() command
        self.h0 = Parameter(torch.zeros(1), requires_grad=False)
        self.c0 = Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, embedded_inputs,
                hidden, mask_length):
        """
        Encoder - Forward-pass

        :param Tensor embedded_inputs: Embedded inputs of Pointer-Net
        :param Tensor hidden: Initiated hidden units for the LSTMs (h, c)
        :return: LSTMs outputs and hidden units (h, c)
        """
        embedded_inputs = embedded_inputs.permute(1, 0, 2)

        # pack the lstm
        # embedded_inputs = torch.nn.utils.rnn.pack_padded_sequence(embedded_inputs, mask_length.cpu(), batch_first=True, enforce_sorted=False)

        # now run through LSTM
        outputs, hidden = self.lstm(embedded_inputs, hidden)
        # print(outputs.batch_sizes)

        # undo the packing operation
        # outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        return outputs.permute(1, 0, 2), hidden

    def init_hidden(self, embedded_inputs):
        """
        Initiate hidden units

        :param Tensor embedded_inputs: The embedded input of Pointer-NEt
        :return: Initiated hidden units for the LSTMs (h, c)
        """

        batch_size = embedded_inputs.size(0)

        # Reshaping (Expanding)
        h0 = self.h0.unsqueeze(0).unsqueeze(0).repeat(self.n_layers,
                                                      batch_size,
                                                      self.hidden_dim)
        c0 = self.h0.unsqueeze(0).unsqueeze(0).repeat(self.n_layers,
                                                      batch_size,
                                                      self.hidden_dim)

        return h0, c0

class Attention(nn.Module):
    """
    Attention model for Pointer-Net
    """

    def __init__(self, input_dim, hidden_dim):
        """
        Initiate Attention

        :param int input_dim: Input's diamention
        :param int hidden_dim: Number of hidden units in the attention
        """

        super(Attention, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.context_linear = nn.Conv1d(input_dim, hidden_dim, 1, 1)
        self.V = Parameter(torch.FloatTensor(hidden_dim), requires_grad=True)
        self._inf = Parameter(torch.FloatTensor([float('-inf')]), requires_grad=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

        # Initialize vector V
        nn.init.uniform(self.V, -1, 1)

    def forward(self, input, context, mask):
        """
        Attention - Forward-pass

        :param Tensor input: Hidden state h
        :param Tensor context: Attention context
        :param ByteTensor mask: Selection mask
        :return: tuple of - (Attentioned hidden state, Alphas)
        """
        # (batch, hidden_dim, seq_len)
        inp = self.input_linear(input).unsqueeze(2).expand(-1, -1, context.size(1))

        # (batch, hidden_dim, seq_len)
        context = context.permute(0, 2, 1)
        ctx = self.context_linear(context)

        # (batch, 1, hidden_dim)
        V = self.V.unsqueeze(0).expand(context.size(0), -1).unsqueeze(1)

        # (batch, seq_len)
        att = torch.bmm(V, self.tanh(inp + ctx)).squeeze(1)
        
        if len(att[mask]) > 0:
            att[mask] = self.inf[mask]
        alpha = self.softmax(att)

        hidden_state = torch.bmm(ctx, alpha.unsqueeze(2)).squeeze(2)

        return hidden_state, alpha

    def init_inf(self, mask_size):
        self.inf = self._inf.unsqueeze(1).expand(*mask_size)

class Decoder(nn.Module):
    """
    Decoder model for Pointer-Net
    """

    def __init__(self, embedding_dim, hidden_dim, output_K):
        """
        Initiate Decoder

        :param int embedding_dim: Number of embeddings in Pointer-Net
        :param int hidden_dim: Number of hidden units for the decoder's RNN
        """

        super(Decoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_K = output_K

        self.input_to_hidden = nn.Linear(embedding_dim, 4 * hidden_dim)
        self.hidden_to_hidden = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.hidden_out = nn.Linear(hidden_dim * 2, hidden_dim)
        self.att = Attention(hidden_dim, hidden_dim)

        # Used for propagating .cuda() command
        self.mask = Parameter(torch.ones(1), requires_grad=False)
        self.runner = Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, embedded_inputs, decoder_input, hidden, context):
        """
        Decoder - Forward-pass

        :param Tensor embedded_inputs: Embedded inputs of Pointer-Net
        :param Tensor decoder_input: First decoder's input
        :param Tensor hidden: First decoder's hidden states
        :param Tensor context: Encoder's outputs
        :return: (Output probabilities, Pointers indices), last hidden state
        """

        batch_size = embedded_inputs.size(0)
        input_length = embedded_inputs.size(1)

        # (batch, seq_len)
        mask = self.mask.repeat(input_length).unsqueeze(0).repeat(batch_size, 1)
        self.att.init_inf(mask.size())

        # Generating arang(input_length), broadcasted across batch_size
        runner = self.runner.repeat(input_length)
        for i in range(input_length):
            runner.data[i] = i
        runner = runner.unsqueeze(0).expand(batch_size, -1).long()

        outputs = []
        pointers = []

        def step(x, hidden):
            """
            Recurrence step function

            :param Tensor x: Input at time t
            :param tuple(Tensor, Tensor) hidden: Hidden states at time t-1
            :return: Hidden states at time t (h, c), Attention probabilities (Alpha)
            """

            # Regular LSTM
            h, c = hidden

            gates = self.input_to_hidden(x) + self.hidden_to_hidden(h)
            input, forget, cell, out = gates.chunk(4, 1)

            input = F.sigmoid(input)
            forget = F.sigmoid(forget)
            cell = F.tanh(cell)
            out = F.sigmoid(out)

            c_t = (forget * c) + (input * cell)
            h_t = out * F.tanh(c_t)

            # Attention section
            hidden_t, output = self.att(h_t, context, torch.eq(mask, 0))
            hidden_t = F.tanh(self.hidden_out(torch.cat((hidden_t, h_t), 1)))

            return hidden_t, c_t, output

        # Recurrence loop
        # for _ in range(input_length):
        for _ in range(self.output_K):
            h_t, c_t, outs = step(decoder_input, hidden)
            hidden = (h_t, c_t)

            # Masking selected inputs
            masked_outs = outs * mask

            # Get maximum probabilities and indices
            max_probs, indices = masked_outs.max(1)
            one_hot_pointers = (runner == indices.unsqueeze(1).expand(-1, outs.size()[1])).float()

            # Update mask to ignore seen indices
            mask  = mask * (1 - one_hot_pointers)

            # Get embedded inputs by max indices
            embedding_mask = one_hot_pointers.unsqueeze(2).expand(-1, -1, self.embedding_dim).byte()
            decoder_input = embedded_inputs[embedding_mask.data].view(batch_size, self.embedding_dim)

            outputs.append(outs.unsqueeze(0))
            pointers.append(indices.unsqueeze(1))
            
        outputs = torch.cat(outputs).permute(1, 0, 2)
        pointers = torch.cat(pointers, 1)

        return (outputs, pointers), hidden

class PointerNet(nn.Module):
    """
    Pointer-Net
    """

    def __init__(self, embedding_dim, hidden_dim, output_K, lstm_layers, dropout, bidir=False):
        """
        Initiate Pointer-Net

        :param int embedding_dim: Number of embbeding channels
        :param int hidden_dim: Encoders hidden units
        :param int lstm_layers: Number of layers for LSTMs
        :param float dropout: Float between 0-1
        :param bool bidir: Bidirectional
        """

        super(PointerNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.bidir = bidir
        self.encoder = Encoder(embedding_dim, hidden_dim, lstm_layers, dropout, bidir)
        # self.embedding = nn.Linear(2, embedding_dim)
        self.decoder = Decoder(embedding_dim, hidden_dim, output_K)
        self.decoder_input0 = Parameter(torch.FloatTensor(embedding_dim), requires_grad=False)

        # Initialize decoder_input0
        nn.init.uniform(self.decoder_input0, -1, 1)

    def forward(self, inputs, mask_length):
        """
        PointerNet - Forward-pass

        :param Tensor inputs: Input sequence
        :return: Pointers probabilities and indices
        """

        batch_size = inputs.size(0)
        input_length = inputs.size(1)

        decoder_input0 = self.decoder_input0.unsqueeze(0).expand(batch_size, -1)

        inputs = inputs.view(batch_size * input_length, -1)
        # embedded_inputs = self.embedding(inputs).view(batch_size, input_length, -1)
        embedded_inputs = inputs.view(batch_size, input_length, -1)

        encoder_hidden0 = self.encoder.init_hidden(embedded_inputs)
        encoder_outputs, encoder_hidden = self.encoder(embedded_inputs,
                                                       encoder_hidden0,mask_length)
        # print("----------PointerNet module----------")
        # print(f"encoder_outputs:{encoder_hidden[1].shape}")
        
        if self.bidir:
            decoder_hidden0 = (torch.cat((encoder_hidden[0][-2:][0],encoder_hidden[0][-2:][1]), dim=-1),
                               torch.cat((encoder_hidden[1][-2:][0],encoder_hidden[1][-2:][1]), dim=-1))
        else:
            decoder_hidden0 = (encoder_hidden[0][-1],
                               encoder_hidden[1][-1])

        (outputs, pointers), decoder_hidden = self.decoder(embedded_inputs,
                                                           decoder_input0,
                                                           decoder_hidden0,
                                                           encoder_outputs)

        return  outputs, pointers

class GPNN_Layer(nn.Module):
    def __init__(self, input_dim=768, output_dim=768, output_K=2, dropout=0.0, num_chosn_neighbors=8):
        super(GPNN_Layer, self).__init__()
        self.dropout = dropout
        self.K = num_chosn_neighbors
        self.output_K=output_K
        self.input_dim = input_dim
        self.conv_1d_1_v = nn.Conv1d(in_channels=input_dim,out_channels=output_dim, kernel_size=self.output_K, padding=0)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.PtrNet = PointerNet(embedding_dim=input_dim,
                hidden_dim=768,
                dropout=self.dropout,
                lstm_layers=2,
                bidir=False,
                output_K=self.output_K)
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.conv_1d_1_v.weight)

    def forward(self, feature, neighbor_index, mask_length, mask):
        neighbor_embeddings = feature.index_select(0,neighbor_index.view(1,-1).squeeze()).view(feature.shape[0], self.K, feature.shape[1])
        neighbor_embeddings = neighbor_embeddings*mask[:,:,:self.input_dim]
        
        #select nodes
        neighbor_embedding = torch.zeros_like(neighbor_embeddings[:,:self.output_K])
        prob, pointer = self.PtrNet(neighbor_embeddings,mask_length)
        for i in range(self.output_K):
            neighbor_embedding[i] = neighbor_embeddings[i].index_select(dim=0,index=pointer[i])
        xrv = neighbor_embedding
        # 1d conv to extract high-level features
        xrv = xrv.permute(0, 2, 1)
        xrv = self.conv_1d_1_v(xrv)
        # xrv = self.max_pool(xrv)
        xrv = xrv.permute(0, 2, 1).squeeze() 
        return xrv, pointer

class Multilayer_GPNN(nn.Module):
    def __init__(self, num_gpnn_layer=1, ego_drop=0.99, dropout=0.0, encodeDim=768, output_K=2, num_chosn_neighbors=8,):
        super(Multilayer_GPNN, self).__init__()
        self.num_gpnn_layer = num_gpnn_layer
        self.ego_drop = ego_drop
        self.dropout = dropout
        self.encodeDim = encodeDim
        self.output_K = output_K
        self.num_chosn_neighbors=num_chosn_neighbors
        self.embed = GCNConv(self.encodeDim, self.encodeDim)
        self.gpnn = GPNN_Layer(self.encodeDim, self.encodeDim, output_K=self.output_K, dropout=self.dropout, num_chosn_neighbors=num_chosn_neighbors,)
        # self.out = nn.Linear(self.encodeDim*3,self.encodeDim)

    def reset_parameters(self):

        self.embed.reset_parameters()
        self.gpnn.reset_parameters()
        # nn.init.xavier_uniform_(self.out.weight)

    def forward(self, contextual_embeddings, edge_index, neighbor_index, mask_length, mask, syntactic_features):
        embeddings = contextual_embeddings
        # embeddings = torch.cat((contextual_embeddings, syntactic_features), dim=-1)
        
        x0 = F.dropout(embeddings, p=self.ego_drop, training=self.training)

        x1 = self.embed(embeddings, edge_index).relu()
        x1 = F.dropout(x1, p=self.dropout, training=self.training)

        # x2, pointer = self.gpnn(x1, neighbor_index, mask_length, mask).relu()
        x2, pointer = self.gpnn(x1, neighbor_index, mask_length, mask)
        x2 = x2.relu()
        x2 = F.dropout(x2, p=self.dropout, training=self.training)

        xf = torch.cat((x0, x1, x2),dim=1)
        # x = self.out(xf)
    
        return xf, pointer

@dataclass(frozen=True)
class GPNNInputFeatures():
    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None
    # sentence_aligned_with_tokenizer: Optional[str] = None
    edge_index: Optional[torch.Tensor] = None
    neighbor_index: Optional[torch.Tensor] = None
    mask_length: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None
    syntactic_features: Optional[torch.Tensor] = None
    
    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"
    
class ChemProtProcessor(DataProcessor):

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train_dev.tsv")), "train")
        # return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        # return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "test")
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3", "4", "5"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        text_index = 1 if set_type == "test" else 0
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[text_index]
            label = None if set_type == "test" else line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

chemprot_processors = {
    "chemprot": ChemProtProcessor,
}
chemprot_output_modes = {
    "chemprot": "classification",
}
chemprot_task_num_labels = {
    "chemprot": 6,
}

def chemprot_convert_examples_to_features(
    examples: Union[List[InputExample], "tf.data.Dataset"],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    task=None,
    label_list=None,
    output_mode=None,
    num_chosn_neighbors=16,
):
    return _chemprot_convert_examples_to_features(
        examples, tokenizer, max_length=max_length, task=task, 
        label_list=label_list, output_mode=output_mode, num_chosn_neighbors=num_chosn_neighbors,
    )

def _chemprot_convert_examples_to_features(
    examples: List[InputExample],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    task=None,
    label_list=None,
    output_mode=None,
    num_chosn_neighbors=16,
    secondary_hoop=True,
):  
    if max_length is None:
        max_length = tokenizer.max_len

    if task is not None:
        processor = chemprot_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = glue_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}
    
    def syntactic_neighbor_generation(sentence, tokens_with_padding, max_length, num_chosn_neighbors, secondary_hoop):
        adjacent_matrix_with_self = torch.zeros((max_length, max_length), dtype = torch.int32)
        adjacent_matrix_without_self = torch.zeros((max_length, max_length), dtype = torch.int32)
        syntactic_features = torch.zeros((max_length, max_length), dtype = torch.float32)
        entity_start_index = [i for i, x in enumerate(tokens_with_padding) if x == "@"]
        entity_end_index = [i for i, x in enumerate(tokens_with_padding) if x == "$"]
        
        # sentence parsing
        sentence_dependency_parse = nlp(sentence)
        pos = []
        tag = []
        dependency = []
        parse_tokens = []
        for token in sentence_dependency_parse:
            pos.append(token.pos_)
            tag.append(token.tag_)
            dependency.append((token.i, token.head.i, token.dep_))
            parse_tokens.append(token.text)
        bool_list = [0] * max_length
        for i,item in enumerate(tokens_with_padding):
            if item[0:2] != "##" and item != "[CLS]" and item != "[SEP]" and item != "[PAD]" and item != "[UNK]":
                bool_list[i] = 1
        word_index = [i for i,x in enumerate(bool_list) if x==1]
        
        # generating adjacent matrix
        if len(entity_start_index) == 2 and len(entity_end_index) == 2:
            neighbor_CLS_index = [x for x in word_index if (int(x) > int(entity_start_index[0]) and int(x) < int(entity_end_index[0])) 
                                  or (int(x) > int(entity_start_index[1]) and int(x) < int(entity_end_index[1]))]
            for neighbor in neighbor_CLS_index:
                adjacent_matrix_with_self[0, int(neighbor)] = 1
                adjacent_matrix_with_self[int(neighbor), 0] = 1
                adjacent_matrix_without_self[0, int(neighbor)] = 1
                adjacent_matrix_without_self[int(neighbor), 0] = 1
        else:
            print(len(entity_start_index), len(entity_end_index))
        for i in range(max_length):
            if tokens_with_padding[i] == "[PAD]":
                pass
            else:
                adjacent_matrix_with_self[i][i] = 1.0
        for i, item in enumerate(tokens_with_padding):
            if item[0:2] == "##":
                adjacent_matrix_with_self[i-1][i-1] = 1.0
                adjacent_matrix_with_self[i][i] = 1.0
                adjacent_matrix_with_self[i][i-1] = 1.0
                adjacent_matrix_with_self[i-1][i] = 1.0
                adjacent_matrix_without_self[i][i-1] = 1.0
                adjacent_matrix_without_self[i-1][i] = 1.0
            else:
                pass
        for tail, head, rel in dependency:
            rel = rel.lower()
            try:
                adjacent_matrix_with_self[word_index[tail]][word_index[tail]] = 1.0
                adjacent_matrix_with_self[word_index[head]][word_index[head]] = 1.0
                adjacent_matrix_with_self[word_index[tail]][word_index[head]] = 1.0
                adjacent_matrix_with_self[word_index[head]][word_index[tail]] = 1.0
                adjacent_matrix_without_self[word_index[tail]][word_index[head]] = 1.0
                adjacent_matrix_without_self[word_index[head]][word_index[tail]] = 1.0
            except:
                print(parse_tokens, word_index)
                print(tokens_with_padding)
        
        #generating neighbor cluster for GPNN
        edge_index = dense_to_sparse(adjacent_matrix_with_self)[0]
        adj_1_hoop = adjacent_matrix_without_self
        adj_2_hoop = ((adjacent_matrix_with_self.float() @ adjacent_matrix_with_self.float())>0).int()-adjacent_matrix_with_self.int()
        neighbor_index = torch.zeros(max_length, num_chosn_neighbors, dtype=int)
        mask_length = torch.zeros(max_length, dtype=int)
        mask = torch.zeros(max_length, num_chosn_neighbors)
        num_chosn_neighbors = num_chosn_neighbors - 1 # center node and k-1 neighbors
        for i in range(max_length):
            if tokens_with_padding[i] == "[PAD]":
                pass
            index1 = torch.nonzero(adj_1_hoop[i])
            num_1_hoop_neighbors = index1.size()[0]
            num_2_hoop_neighbors = num_chosn_neighbors - num_1_hoop_neighbors
            neighbor_index[i,0]=i
            mask_length[i] = min(num_1_hoop_neighbors+1,num_chosn_neighbors+1)
            # 1 hop
            for j1 in range(num_1_hoop_neighbors):
                if j1>=num_chosn_neighbors: break
                neighbor_index[i,j1+1]=index1[j1]
            # 2 hop
            if secondary_hoop and num_2_hoop_neighbors>0:
                index2 = torch.nonzero(adj_2_hoop[i])
                mask_length[i] = min(num_1_hoop_neighbors+index2.size()[0]+1,num_chosn_neighbors+1)
                for j2 in range(index2.size()[0]):
                    if j2>=num_2_hoop_neighbors: break
                    neighbor_index[i,j2+num_1_hoop_neighbors+1]=index2[j2]
            mask[i] = torch.tensor([1]*mask_length[i]+[0]*(num_chosn_neighbors+1-mask_length[i]))
            # if i<10:
            #     print(f"Node:{i}   neighbor_index:{neighbor_index[i]}  mask_length:{mask_length[i]}")
                
        mask = mask.unsqueeze(2).repeat(1,1,768)
        
        return edge_index, neighbor_index, mask_length, mask, syntactic_features   
    
    def label_from_example(example: InputExample): ## -> Union[int, float, None]
        if example.label is None:
            return None
        if output_mode == "classification":
            return label_map[example.label]
        elif output_mode == "regression":
            return float(example.label)
        raise KeyError(output_mode)

    labels = [label_from_example(example) for example in examples]
    batch_encoding = tokenizer(
        [(example.text_a, example.text_b) for example in examples],
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )
        
    features = []
    # for i in range(len(examples)):
    for i in range(500):
        print(i)
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}
        tokens_with_padding = tokenizer.convert_ids_to_tokens(inputs["input_ids"])
        sentence_aligned_with_tokenizer = ""
        for token in tokens_with_padding:
            if token == "[PAD]" or token == "[CLS]" or token == "[SEP]" or token == "[UNK]":
                pass
            elif "##" in token:
                sentence_aligned_with_tokenizer += token[2:]
            else:
                sentence_aligned_with_tokenizer += " "
                sentence_aligned_with_tokenizer += token
        sentence_aligned_with_tokenizer = sentence_aligned_with_tokenizer[1:]
        edge_index, neighbor_index, mask_length, mask, syntactic_features = syntactic_neighbor_generation(
            sentence_aligned_with_tokenizer, tokens_with_padding, max_length, num_chosn_neighbors, secondary_hoop)
        feature = GPNNInputFeatures(
            input_ids = inputs["input_ids"],
            token_type_ids = inputs["token_type_ids"],
            attention_mask = inputs["attention_mask"],
            label = labels[i],
            edge_index = edge_index,
            neighbor_index = neighbor_index,
            mask_length = mask_length,
            mask = mask,
            syntactic_features = syntactic_features,
        )
        features.append(feature)

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("features: %s" % features[i])

    return features

@dataclass
class ChemProtDataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    task_name: str = field(metadata={"help": "The name of the task to train on: " + "chemprot"})
    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    num_chosn_neighbors: int = field(
        default=16, metadata={"help": "Number of chosn neighbors in Graph pointer neural network"}
    )
    num_GPNN_output_node: int = field(
        default=4, metadata={"help": "Number of output neighbors in Graph pointer neural network"}
    )
    def __post_init__(self):
        self.task_name = self.task_name.lower()

class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"

class ChemProtDataset(Dataset):
    args: ChemProtDataTrainingArguments
    output_mode: str
    features: List[GPNNInputFeatures]
    def __init__(
        self,
        args: ChemProtDataTrainingArguments,
        tokenizer: PreTrainedTokenizer,
        limit_length: Optional[int] = None,
        mode: Union[str, Split] = Split.train,
        cache_dir: Optional[str] = None,
    ):
        self.args = args
        self.processor = chemprot_processors[args.task_name]()
        self.output_mode = chemprot_output_modes[args.task_name]
        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError("mode is not a valid split name")
        # Load data features from cache or dataset file
        extra_mode = "linking_CLS_with_entities"
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else args.data_dir,
            "cached_{}_{}_{}_{}_{}_{}_{}".format(
                mode.value, tokenizer.__class__.__name__, str(args.max_seq_length), str(args.num_chosn_neighbors), str(args.num_GPNN_output_node), args.task_name, extra_mode,
            ),
        )
        label_list = self.processor.get_labels()
        self.label_list = label_list

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                start = time.time()
                self.features = torch.load(cached_features_file)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {args.data_dir}")

                if mode == Split.dev:
                    examples = self.processor.get_dev_examples(args.data_dir)
                elif mode == Split.test:
                    examples = self.processor.get_test_examples(args.data_dir)
                else:
                    examples = self.processor.get_train_examples(args.data_dir)
                if limit_length is not None:
                    examples = examples[:limit_length]
                self.features = chemprot_convert_examples_to_features(
                    examples,
                    tokenizer,
                    max_length=args.max_seq_length,
                    label_list=label_list,
                    output_mode=self.output_mode,
                    num_chosn_neighbors=args.num_chosn_neighbors,
                )
                start = time.time()
                torch.save(self.features, cached_features_file)
                # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i]

    def get_labels(self):
        return self.label_list

class SequenceClassifierOutput(ModelOutput):
    """
    Base class for outputs of sentence classification models.
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    warmup_proportion: Optional[float] = field(
        default=0.1, metadata={"help": "Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% of training."}
    )
    classifier_dropout: Optional[float] = field(
        default=0.1, metadata={"help": "Proportion of dropout rate for classification process. E.g., 0.1 = 10% of classification."}
    )

class GELUActivation(nn.Module):
    def __init__(self, use_gelu_python: bool = False):
        super().__init__()
        if use_gelu_python:
            self.act = self._gelu_python
        else:
            self.act = nn.functional.gelu

    def _gelu_python(self, input):
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))

    def forward(self, input):
        return self.act(input)

ACT2FN = {
    "gelu": GELUActivation(),
    "gelu_python": GELUActivation(use_gelu_python=True),
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
}

def get_activation(activation_string):
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError(f"function {activation_string} not found in ACT2FN mapping {list(ACT2FN.keys())}")

class ElectraforRelationClassificationConfig(ElectraConfig):
    def __init__(self, num_labels=0, classifier_dropout=0.0, num_GPNN_output_node=2, num_chosn_neighbors=8, **kwargs):
        self.num_labels = num_labels
        self.classifier_dropout = classifier_dropout
        self.num_GPNN_output_node = num_GPNN_output_node
        self.num_chosn_neighbors = num_chosn_neighbors
        super().__init__(**kwargs)

class RelationClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = get_activation("gelu")(x)  # although BERT uses tanh here, it seems Electra authors used gelu here
        # x = nn.Tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
    
class GPNNbasedRelationClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense_mid = nn.Linear(128, 1)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        use_CLS = False
        if use_CLS:
            x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        else:
            x = features.permute(0, 2, 1)
            x = self.dense_mid(x)
            x = get_activation("gelu")(x)
            x = x.permute(0, 2, 1)
            x = features[:, 0, :]
        
        x = self.dropout(x)
        x = self.dense(x)
        x = get_activation("gelu")(x)  # although BERT uses tanh here, it seems Electra authors used gelu here
        # x = nn.Tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class ElectraForRelationClassification(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.electra = ElectraModel(config)
        self.classifier = GPNNbasedRelationClassificationHead(config)
        self.num_chosn_neighbors = config.num_chosn_neighbors
        self.output_K = config.num_GPNN_output_node
        self.gpnn = Multilayer_GPNN(output_K=self.output_K, num_chosn_neighbors=self.num_chosn_neighbors,)
        self.gpnn_second_layer_input_input = Multilayer_GPNN(output_K=self.output_K, num_chosn_neighbors=self.num_chosn_neighbors,)
        self.gpnn_second_layer_output_output = Multilayer_GPNN(output_K=self.output_K, num_chosn_neighbors=self.num_chosn_neighbors,)
        self.dense = nn.Linear(3*config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(0.0)
        self.init_weights()
        # self.post_init()
        
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        sentence_aligned_with_tokenizer: Optional[List[str]] = None,
        edge_index: Optional[torch.Tensor] = None,
        neighbor_index: Optional[torch.Tensor] = None,
        mask_length: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        syntactic_features: Optional[torch.Tensor] = None,
    ):
        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = discriminator_hidden_states[0]
        edge_index = torch.tensor(edge_index, dtype=torch.int64)
        
        batch_size = sequence_output.size(0)
        gpnn_output_tuple = ()
        gpnn_pointer_tuple = ()
        
        input_input = False
        output_output = False

        if input_input:
            gpnn_input_output_tuple = ()
            gpnn_input_pointer_tuple = ()
            for _ in range(batch_size):
                gpnn_output_input, pointer_input = self.gpnn_second_layer_input_input(sequence_output[_], edge_index[_], neighbor_index[_], mask_length[_], mask[_], syntactic_features[_])
                gpnn_input_output_tuple = gpnn_input_output_tuple + (gpnn_output_input,)
                gpnn_input_pointer_tuple = gpnn_input_pointer_tuple + (pointer_input,)
            gpnn_input_outputs = torch.stack(gpnn_input_output_tuple)
            gpnn_input_pointer = torch.stack(gpnn_input_pointer_tuple)
            sequence_output = gpnn_input_outputs
            sequence_output = self.dense(sequence_output)
            sequence_output = self.dropout(sequence_output)
            
        for _ in range(batch_size):
            gpnn_output, pointer = self.gpnn(sequence_output[_], edge_index[_], neighbor_index[_], mask_length[_], mask[_], syntactic_features[_])
            gpnn_output_tuple = gpnn_output_tuple + (gpnn_output,)
            gpnn_pointer_tuple = gpnn_pointer_tuple + (pointer,)
        gpnn_outputs = torch.stack(gpnn_output_tuple)
        gpnn_pointer = torch.stack(gpnn_pointer_tuple)
        gpnn_outputs = self.dense(gpnn_outputs)
        
        if output_output:
            gpnn_outputs = self.dropout(gpnn_outputs)
            gpnn_output_output_tuple = ()
            gpnn_output_pointer_tuple = ()
            for _ in range(batch_size):
                gpnn_output_output, pointer_output = self.gpnn_second_layer_output_output(gpnn_outputs[_], edge_index[_], neighbor_index[_], mask_length[_], mask[_], syntactic_features[_])
                gpnn_output_output_tuple = gpnn_output_output_tuple + (gpnn_output_output,)
                gpnn_output_pointer_tuple = gpnn_output_pointer_tuple + (pointer_output,)
            gpnn_output_outputs = torch.stack(gpnn_output_output_tuple)
            gpnn_output_pointer = torch.stack(gpnn_output_pointer_tuple)
            gpnn_outputs = gpnn_output_outputs
            gpnn_pointer = gpnn_output_pointer
            gpnn_outputs = self.dense(gpnn_outputs)
        
        logits = self.classifier(gpnn_outputs)

        # outputs = (logits,) + discriminator_hidden_states[1:]  # add hidden states and attention if they are here
        outputs = (logits,)

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))  
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        outputs = outputs + (gpnn_pointer,)

        return outputs # (loss), logits, (hidden_states), (attentions)

InputDataClass = NewType("InputDataClass", Any)

class DataCollatorMixin:
    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        if return_tensors == "tf":
            return self.tf_call(features)
        elif return_tensors == "pt":
            return self.torch_call(features)
        elif return_tensors == "np":
            return self.numpy_call(features)
        else:
            raise ValueError(f"Framework '{return_tensors}' not recognized!")

def gpnn_data_collator(features: List[InputDataClass], return_tensors="pt"):
    if return_tensors == "pt":
        return torch_gpnn_data_collator(features)

@dataclass
class GPNNDataCollator(DataCollatorMixin):
    return_tensors: str = "pt"
    def __call__(self, features: List[Dict[str, Any]], return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        return gpnn_data_collator(features, return_tensors)

def torch_gpnn_data_collator(features: List[InputDataClass]):
    import torch
    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    global max_length_edge
    for feature in features:
        feature["edge_index"] = torch.cat((feature["edge_index"], torch.zeros(2, max_length_edge - feature["edge_index"].size(1))), 1)
    batch = {}
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None:
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, str):
                batch[k] = []
                for f in features:
                    batch[k].append(f[k])
            else:
                batch[k] = torch.tensor([f[k] for f in features])

    return batch

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, ChemProtDataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )

    # Set seed
    set_seed(training_args.seed)

    try:
        if data_args.task_name == "chemprot":
            num_labels = chemprot_task_num_labels[data_args.task_name]
            output_mode = chemprot_output_modes[data_args.task_name]
        else:
            num_labels = glue_tasks_num_labels[data_args.task_name]
            output_mode = glue_output_modes[data_args.task_name]
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))


    # Load tokenizer
    tokenizer = ElectraTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    # Get datasets
    if data_args.task_name == "chemprot":
        train_dataset = (
            ChemProtDataset(data_args, tokenizer=tokenizer, mode="train",cache_dir=model_args.cache_dir) if training_args.do_train else None
            )
        eval_dataset = (
            ChemProtDataset(data_args, tokenizer=tokenizer, mode="dev", cache_dir=model_args.cache_dir)
            if training_args.do_eval
            else None
            )
        test_dataset = (
            ChemProtDataset(data_args, tokenizer=tokenizer, mode="test", cache_dir=model_args.cache_dir)
            if training_args.do_predict
            else None
            )
    else:
        train_dataset = (
            GlueDataset(data_args, tokenizer=tokenizer, cache_dir=model_args.cache_dir) if training_args.do_train else None
            )
        eval_dataset = (
            GlueDataset(data_args, tokenizer=tokenizer, mode="dev", cache_dir=model_args.cache_dir)
            if training_args.do_eval
            else None
            )
        test_dataset = (
            GlueDataset(data_args, tokenizer=tokenizer, mode="test", cache_dir=model_args.cache_dir)
            if training_args.do_predict
            else None
            )
    global max_length_edge
    max_length_edge = 0
    if training_args.do_train:
        for i in range(len(train_dataset)):
            if train_dataset[i].edge_index.size(1) > max_length_edge:
                max_length_edge = train_dataset[i].edge_index.size(1)
    if training_args.do_predict:
        for i in range(len(test_dataset)):
            if test_dataset[i].edge_index.size(1) > max_length_edge:
                max_length_edge = test_dataset[i].edge_index.size(1)

    # Load pretrained model
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    
    # Currently, this code do not support distributed training.
    if training_args.do_train:
        training_args.warmup_steps = int(model_args.warmup_proportion * (len(train_dataset) / training_args.per_device_train_batch_size) * training_args.num_train_epochs)
    training_args_weight_decay = 0.01
    logger.info("Training/evaluation parameters %s", training_args) 
    
    config = ElectraforRelationClassificationConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        num_labels = num_labels,
        classifier_dropout = model_args.classifier_dropout,
        num_GPNN_output_node = data_args.num_GPNN_output_node,
        num_chosn_neighbors = data_args.num_chosn_neighbors,
    )
    
    model = ElectraForRelationClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config = config,
        cache_dir = model_args.cache_dir,
    )
    
    def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            if output_mode == "classification":
                preds = np.argmax(p.predictions, axis=1)
            elif output_mode == "regression":
                preds = np.squeeze(p.predictions)
            return compute_metrics(task_name, preds, p.label_ids)

        return compute_metrics_fn
 
    # Initialize our Trainer
    if data_args.task_name == "chemprot":
        trainer = Trainer_GPNN(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=gpnn_data_collator,
            compute_metrics=build_compute_metrics_fn(data_args.task_name),
        )
    else:
        trainer = Trainer_GPNN(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=build_compute_metrics_fn(data_args.task_name),
        )
    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        # if trainer.is_world_master():
        #     tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
            eval_datasets.append(
                GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="dev", cache_dir=model_args.cache_dir)
            )

        for eval_dataset in eval_datasets:
            trainer.compute_metrics = build_compute_metrics_fn(eval_dataset.args.task_name)
            eval_result = trainer.evaluate(eval_dataset=eval_dataset)

            output_eval_file = os.path.join(
                training_args.output_dir, f"eval_results_{eval_dataset.args.task_name}.txt"
            )
            if trainer.is_world_master():
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Eval results {} *****".format(eval_dataset.args.task_name))
                    for key, value in eval_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))

            eval_results.update(eval_result)

    if training_args.do_predict:
        logging.info("*** Test ***")
        test_datasets = [test_dataset]
        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
            test_datasets.append(
                GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="test", cache_dir=model_args.cache_dir)
            )

        for test_dataset in test_datasets:
            input_ids = [feature.input_ids for feature in test_dataset.features]
            tokens_with_padding = [tokenizer.convert_ids_to_tokens(input_id) for input_id in input_ids]
            test_outputs = trainer.predict(test_dataset=test_dataset)
            predictions = test_outputs.predictions
            pointers = test_outputs.pointers
            ourput_pointers = [pointer[0] for pointer in pointers]
            point_words = [] #significant word list according to the output of GPNN
            for i in range(len(pointers)):
                point_word = [tokens_with_padding[i][int(index)] for index in pointers[i][0]]
                point_words.append(point_word)

            if output_mode == "classification":
                predictions = np.argmax(predictions, axis=1)

            output_test_file = os.path.join(
                training_args.output_dir, 
                f"test_results.txt"
                #f"test_results_{test_dataset.args.task_name}.txt"
            )
            if trainer.is_world_process_zero():
                with open(output_test_file, "w") as writer:
                    logger.info("***** Test results {} *****".format(test_dataset.args.task_name))
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if output_mode == "regression":
                            writer.write("%d\t%3.3f\n" % (index, item))
                        else:
                            item = test_dataset.get_labels()[item]
                            writer.write("%d\t%s\n" % (index, item))
                            
            output_test_significant_words_file = os.path.join(
                training_args.output_dir, 
                f"test_results_significant_words.txt"
                #f"test_results_{test_dataset.args.task_name}.txt"
            )
            if trainer.is_world_process_zero():
                with open(output_test_significant_words_file, "w") as writer:
                    logger.info("***** Test results significant_words {} *****".format(test_dataset.args.task_name))
                    writer.write("index\tsignificant_words\n")
                    for index, item in enumerate(point_words):
                        if output_mode == "regression":
                            writer.write("%d\t%3.3f\n" % (index, item))
                        else:
                            writer.write("%d\t%s\n" % (index, item))
            
            output_test_pointers_and_tokens_file = os.path.join(
                training_args.output_dir, 
                f"test_results_pointers_and_tokens.txt"
                #f"test_results_{test_dataset.args.task_name}.txt"
            )
            if trainer.is_world_process_zero():
                with open(output_test_pointers_and_tokens_file, "w") as writer:
                    logger.info("***** Test results pointers and tokens file {} *****".format(test_dataset.args.task_name))
                    writer.write("index\ttokens\tpointers\n")
                    for index, item in enumerate(ourput_pointers):
                        if output_mode == "regression":
                            writer.write("%d\t%3.3f\n" % (index, item))
                        else:
                            writer.write("%d\t%s\t%s\n" % (index, tokens_with_padding[index], item))
    return eval_results

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

if __name__ == "__main__":
    main()