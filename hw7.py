import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import random_split


from tqdm import tqdm

import importlib

from datetime import datetime as dt
import time

import imdb_voc



root = './'

# import sentences
importlib.reload(imdb_voc)

# set device
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


"""

You can implement any necessary methods.

"""

class MultiHeadAttention(nn.Module):
    '''
    - d_model(int) : Transformer에서의 feature vector의 size
    - d_Q, d_K, d_V(int) : size of Q, K, V for each head of the multi-head attention.
                           Typically passed as (d_model / numhead) from TF_Encoder_Block
    - numhead(int) : Multi-head attention에서 head의 개수
    - dropout(float) : Dropout probaility
    '''
    def __init__(self, d_model, d_Q, d_K, d_V, numhead, dropout):    
      super().__init__()
      
      self.numhead=numhead
      # input linear layers for V, Q, K
      # d_Q, d_K, d_V are typically set to d_model/numhead
      
      self.V_Linear = nn.Linear(in_features=d_model, out_features=d_model)
      self.Q_Linear = nn.Linear(in_features=d_model, out_features=d_model)
      self.K_Linear = nn.Linear(in_features=d_model, out_features=d_model)

      # output linear layer
      self.MHA_Linear = nn.Linear(in_features=d_model, out_features=d_model)
      
      # dropout
      self.dropout=nn.Dropout(dropout)
      
    
    def forward(self, x_Q, x_K, x_V, src_batch_lens=None):
      # This method computes the scaled dot-product attention.
      '''
      1. x_Q, x_K, x_V(tensor) : Q, K, V inputs having shape (B, T_T, d_model), (B, T_S, d_model) and
                                (B, T_S, d_model) respectively.
        * B: batch size / T_S: source sequence length / T_T: target sequence length
      2. src_batch_lens(tensor) : shape=(B, ), contains the length information of batched source.
        만약 batch size가 3이고 input data(token of words from review)가 차례대로 길이가 3, 8, 5라면,
        src_batch_lens는 [3,8,5]가 된다.
        만약, T_S=10이라고 치면, batch의 first input의 길이가 3이었으므로 3 word token과 7 <PAD> token을 가진다.
        Note these <PAD> tokens should be ignored when we compute the attention coefficients. That is, the attention
        coefficient computed from softmax operation should be sufficient small on input positions with <PAD>.
        ***Use src_batch_lens to find out which part of source input is <PAD>!***
      '''
      # Q2. Implement
      # out: tensor, shape=(B, T_T, d_model)
      # Operation
      '''
      - Inputs x_Q, x_K, x_V is first projected to each head through linear layers (Fig 2)
      Then the following Scaled Dot-Product Attention is Fig 2 is applied "to each head".
      (uses Matmul, Scale, Softmax)
      - Mask(opt.) layer masks out source tokens which are <PAD>, so that they have negligible effect on computing
      softmax. The masking can be achieved using "src_batch_lens"
      - Where to put dropout?
        (1) Before applying attention coefficients to V. (softmax까지 계산하고 V랑 곱하기 전에)
        (2) After applying the final Linear layer => 즉, right before returning 'out', apply dropout.
      '''
      d_k = x_K.shape[-1]
      
      attention_score = torch.matmul(x_Q, x_K.transpose(-2, -1))
      attention_score = attention_score / math.sqrt(d_k)

      # print(attention_score.shape) # (256, 384, 384)

      # mask 적용하기
      res = attention_score.clone()
      res[attention_score=='<PAD>'] = 1e-21
      attention_score = res


      # softmax 적용하기
      attention_prob = F.softmax(attention_score, dim=-1)
      attention_prob = self.dropout(attention_prob)

      # V와 최종적으로 matmul
      out = torch.matmul(attention_prob, x_V)
      out = self.dropout(out)

      return out

class TF_Encoder_Block(nn.Module):
    '''
    - d_model(int) : Transformer에서의 feature vector의 size
    - d_ff(int) : Feed Forward block의 feature vector의 size
    - numhead(int) : Multi-head attention에서 head의 개수
    - dropout(float) : Dropout probaility
    '''
    def __init__(self, d_model, d_ff, numhead, dropout):
      # Q3. Implment constructor for transformer encoder block    
      super().__init__()
      
      self.self_attn_layer_norm = nn.LayerNorm(d_model)
      self.ff_layer_norm = nn.LayerNorm(d_model)
      self.self_attention = MultiHeadAttention(d_model, d_model/numhead, d_model/numhead, d_model/numhead, numhead, dropout)
      
      self.Linear1 = nn.Linear(in_features=d_model, out_features=d_ff)
      self.Linear2 = nn.Linear(in_features=d_ff, out_features=d_model)
      self.dropout = nn.Dropout(dropout)

      self.feedforward_block = nn.Sequential(self.Linear1, nn.ReLU(), self.dropout, self.Linear2, self.dropout)
      

    def forward(self, x, src_batch_lens):
        '''
        - d model :int, size of feature vector in the Transformer.
        - x :tensor, x input feature having shape (B, T_S, d model).
        - src batch lens :Same as explained previously. You should pass src batch lens
                          to your MultiHeadAttention object instantiated in this class.
        '''
      
        # Q4. Implment forward function for transformer encoder block
        '''
        [Operation]
        forward function should perform:
        1. Feed input x to multi-head attention layer. Note that the operation of this layer
           is self-attention, so set the input properly.
        2. attention output is added to x (skip connection), then perform layer normalization
        3. then the output is fed into feed forward layer
        4. feed forward output is added to its input (skip connection), then perform layer
           normalization
        '''

        mha_out = self.self_attention.forward(x, x, x, src_batch_lens)
        layernorm_out1 = self.self_attn_layer_norm(x + mha_out)
        ff_out = self.feedforward_block(layernorm_out1)
        out = self.ff_layer_norm(layernorm_out1 + ff_out)

        return out

"""
Positional encoding
PE(pos,2i) = sin(pos/10000**(2i/dmodel))
PE(pos,2i+1) = cos(pos/10000**(2i/dmodel))
"""

def PosEncoding(t_len, d_model):
    i = torch.tensor(range(d_model))
    pos = torch.tensor(range(t_len))
    POS, I = torch.meshgrid(pos, i)
    PE = (1-I % 2)*torch.sin(POS/10**(4*I/d_model)) + (I%2)*torch.cos(POS/10**(4*(I-1)/d_model))
    return PE

class TF_Encoder(nn.Module):
    def __init__(self, vocab_size, d_model,
                 d_ff, numlayer, numhead, dropout):    
        super().__init__()
        
        self.numlayer = numlayer
        self.src_embed  = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.dropout=nn.Dropout(dropout)

        '''
        - vocab size, :int, size of vocabulary, i.e., the total number of words recognized
                       by the model.
        - d model :int, size of feature vector in the Transformer.
        - d ff :int, size of feature vector in Feed Forward block.
        - numlayer :int, number of TF Encoder Block in the encoder (N in Fig. 1)
        - numhead :int, number of heads in multi-head attention.
        - dropout :float, dropout probability.
        '''

        # Q5. Implement a sequence of numlayer encoder blocks
        self.layers = nn.ModuleList([TF_Encoder_Block(d_model, d_ff, numhead, dropout) for _ in range(numlayer)])
        
    def forward(self, x, src_batch_lens):

      x_embed = self.src_embed(x)
      x = self.dropout(x_embed)
      p_enc = PosEncoding(x.shape[1], x.shape[2]).to(dev)
      x = x + p_enc
        
      # Q6. Implement: forward over numlayer encoder blocks
      '''
      - x :tensor, a batch of input tokens having shape (B,T_S). B is batch size, T_S is
           the sequence length of tokens. Regardless of the length of review words in each
           batch, each batch is padded to length T_S.
      - src batch lens :Same as explained previously. You should pass src batch lens
                        to your MultiHeadAttention object instantiated in this class.
      '''
      for layer in self.layers:
        x = layer(x, src_batch_lens)
      
      return x



"""

main model

"""

class sentiment_classifier(nn.Module):
    
    def __init__(self, enc_input_size, 
                 enc_d_model,
                 enc_d_ff,
                 enc_num_layer,
                 enc_num_head,
                 dropout,
                ):    
        super().__init__()
        
        self.encoder = TF_Encoder(vocab_size = enc_input_size,
                                  d_model = enc_d_model, d_ff=enc_d_ff,
                                  numlayer=enc_num_layer, numhead=enc_num_head,
                                  dropout=dropout)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,None)),
            nn.Dropout(dropout),
            nn.Linear(in_features = enc_d_model, out_features=enc_d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features = enc_d_model, out_features = 1)
        )
          
   
    def forward(self, x, x_lens):
        src_ctx = self.encoder(x, src_batch_lens = x_lens)
        # size should be (b,)
        out_logits = self.classifier(src_ctx).flatten()

        return out_logits

"""

datasets

"""

# Load IMDB dataset
# once you build the dataset, you can load it from file to save time
# to load from file, set this flag True
load_imdb_dataset = True

if load_imdb_dataset:
    imdb_dataset = torch.load('imdb_dataset.pt')
else:
    imdb_dataset = imdb_voc.IMDB_tensor_dataset()
    torch.save(imdb_dataset, 'imdb_dataset.pt')

train_dataset, test_dataset = imdb_dataset.get_dataset()

split_ratio = 0.85
num_train = int(len(train_dataset) * split_ratio)
split_train, split_valid = random_split(train_dataset, [num_train, len(train_dataset) - num_train])

# Set hyperparam (batch size)
batch_size_trn = 256
batch_size_val = 256
batch_size_tst = 256

train_dataloader = DataLoader(split_train, batch_size=batch_size_trn, shuffle=True)
val_dataloader = DataLoader(split_valid, batch_size=batch_size_val, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size_tst, shuffle=True)

# get character dictionary
src_word_dict = imdb_dataset.src_stoi
src_idx_dict = imdb_dataset.src_itos

SRC_PAD_IDX = src_word_dict['<PAD>']

# show sample reviews with pos/neg sentiments

show_sample_reviews = True

if show_sample_reviews:
    
    sample_text, sample_lab = next(iter(train_dataloader))
    slist=[]

    for stxt in sample_text[:4]: 
        slist.append([src_idx_dict[j] for j in stxt])

    for j, s in enumerate(slist):
        print('positive' if sample_lab[j]==1 else 'negative')
        print(' '.join([i for i in s if i != '<PAD>'])+'\n')


"""

model

"""

enc_vocab_size = len(src_word_dict) # counting eof, one-hot vector goes in

# Set hyperparam (model size)
# examples: model & ff dim - 8, 16, 32, 64, 128, numhead, numlayer 1~4

enc_d_model = 16

enc_d_ff = 16

enc_num_head = 4

enc_num_layer= 4

DROPOUT=0.1

model = sentiment_classifier(enc_input_size=enc_vocab_size,
                         enc_d_model = enc_d_model,     
                         enc_d_ff = enc_d_ff, 
                         enc_num_head = enc_num_head, 
                         enc_num_layer = enc_num_layer,
                         dropout=DROPOUT) 

model = model.to(dev)

"""

optimizer

"""

# Set hyperparam (learning rate)
# examples: 1e-3 ~ 1e-5

lr = 1e-3

optimizer = torch.optim.Adam(model.parameters(), lr = lr)

criterion = nn.BCEWithLogitsLoss()

"""

auxiliary functions

"""


# get length of reviews in batch
def get_lens_from_tensor(x):
    # lens (batch, t)
    lens = torch.ones_like(x).long()
    lens[x==SRC_PAD_IDX]=0
    return torch.sum(lens, dim=-1)

def get_binary_metrics(y_pred, y):
    # find number of TP, TN, FP, FN
    TP=sum(((y_pred == 1)&(y==1)).type(torch.int32))
    FP=sum(((y_pred == 1)&(y==0)).type(torch.int32))
    TN=sum(((y_pred == 0)&(y==0)).type(torch.int32))
    FN=sum(((y_pred == 0)&(y==1)).type(torch.int32))
    accy = (TP+TN)/(TP+FP+TN+FN)
            
    recall = TP/(TP+FN) if TP+FN!=0 else 0
    prec = TP/(TP+FP) if TP+FP!=0 else 0
    f1 = 2*recall*prec/(recall+prec) if recall+prec !=0 else 0
    
    return accy, recall, prec, f1

"""

train/validation

""" 


def train(model, dataloader, optimizer, criterion, clip):

    model.train()

    epoch_loss = 0

    for i, batch in enumerate(dataloader):

        src = batch[0].to(dev)
        trg = batch[1].float().to(dev)

        # print('batch trg.shape', trg.shape)
        # print('batch src.shape', src.shape)

        optimizer.zero_grad()

        x_lens = get_lens_from_tensor(src).to(dev)

        output = model(x=src, x_lens=x_lens) 


        output = output.contiguous().view(-1)
        trg = trg.contiguous().view(-1)
        
        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

def evaluate(model, dataloader, criterion):

    model.eval()
    
    epoch_loss = 0
    
    epoch_accy =0
    epoch_recall =0
    epoch_prec =0
    epoch_f1 =0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):

            src = batch[0].to(dev)
            trg = batch[1].float().to(dev)

            x_lens = get_lens_from_tensor(src).to(dev)

            output = model(x=src, x_lens=x_lens) 

            output = output.contiguous().view(-1)
            trg = trg.contiguous().view(-1)

            loss = criterion(output, trg)
            
            accy, recall, prec, f1 = get_binary_metrics((output>=0).long(), trg.long())
            epoch_accy += accy
            epoch_recall += recall
            epoch_prec += prec
            epoch_f1 += f1

            epoch_loss += loss.item()

    # show accuracy
    print(f'\tAccuracy: {epoch_accy/(len(dataloader)):.3f}')
    
    return epoch_loss / len(dataloader)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

"""

Training loop

"""

N_EPOCHS = 30
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss = train(model, train_dataloader, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, val_dataloader, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'model.pt')

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f}')
        
"""

Test loop

"""
print('*** Now test phase begins! ***')
model.load_state_dict(torch.load('model.pt'))

test_loss = evaluate(model, test_dataloader, criterion)

print(f'| Test Loss: {test_loss:.3f}')