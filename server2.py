import pandas as pd

import torch
import torch.nn as nn

import math
import download
import pickle

import random
max_seq_len=34
pd.set_option('display.max_colwidth', None)

print("here")



# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask, request, jsonify
import json




#creating model template
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=max_seq_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if self.pe.size(0) < x.size(0):
            self.pe = self.pe.repeat(x.size(0), 1, 1)
        self.pe = self.pe[:x.size(0), :, :]

        x = x + self.pe
        return self.dropout(x)


class ImageCaptionModel(nn.Module):
    def __init__(self, n_head, n_decoder_layer, vocab_size, embedding_size):
        super(ImageCaptionModel, self).__init__()
        self.pos_encoder = PositionalEncoding(embedding_size, 0.1)
        self.TransformerDecoderLayer = nn.TransformerDecoderLayer(d_model=embedding_size, nhead=n_head)
        self.TransformerDecoder = nn.TransformerDecoder(decoder_layer=self.TransformerDecoderLayer,
                                                        num_layers=n_decoder_layer)
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.last_linear_layer = nn.Linear(embedding_size, vocab_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.last_linear_layer.bias.data.zero_()
        self.last_linear_layer.weight.data.uniform_(-initrange, initrange)

    def generate_Mask(self, size, decoder_inp):
        decoder_input_mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        decoder_input_mask = decoder_input_mask.float().masked_fill(decoder_input_mask == 0, float('-inf')).masked_fill(
            decoder_input_mask == 1, float(0.0))

        decoder_input_pad_mask = decoder_inp.float().masked_fill(decoder_inp == 0, float(0.0)).masked_fill(
            decoder_inp > 0, float(1.0))
        decoder_input_pad_mask_bool = decoder_inp == 0

        return decoder_input_mask, decoder_input_pad_mask, decoder_input_pad_mask_bool

    def forward(self, encoded_image, decoder_inp):
        encoded_image = encoded_image.permute(1, 0, 2)

        decoder_inp_embed = self.embedding(decoder_inp) * math.sqrt(self.embedding_size)

        decoder_inp_embed = self.pos_encoder(decoder_inp_embed)
        decoder_inp_embed = decoder_inp_embed.permute(1, 0, 2)

        decoder_input_mask, decoder_input_pad_mask, decoder_input_pad_mask_bool = self.generate_Mask(
            decoder_inp.size(1), decoder_inp)
        decoder_input_mask = decoder_input_mask
        decoder_input_pad_mask = decoder_input_pad_mask
        decoder_input_pad_mask_bool = decoder_input_pad_mask_bool

        decoder_output = self.TransformerDecoder(tgt=decoder_inp_embed, memory=encoded_image,
                                                 tgt_mask=decoder_input_mask,
                                                 tgt_key_padding_mask=decoder_input_pad_mask_bool)

        final_output = self.last_linear_layer(decoder_output)

        return final_output, decoder_input_pad_mask



# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__)


# @app.route('/init')
# def function_to_run_only_once():
    # loading pickle data

dbfile = open('index_to_word', 'rb')
index_to_word = pickle.load(dbfile)
print('loading indextoword')
dbfile.close()

dbfile = open('word_to_index', 'rb')
word_to_index = pickle.load(dbfile)
print('loading wordtoindex')
dbfile.close()

# download model from google driver
download.download_from_drive()
print('downloading model')

## Generate Captions !!!
model = ImageCaptionModel(16, 4, 8812, 512)
model.load_state_dict(torch.load("model_state.pth", map_location=torch.device('cpu')) )
model.eval()
# model = torch.load('./BestModel1', map_location=torch.device('cpu'))
print('loading model')
start_token = word_to_index['<start>']
end_token = word_to_index['<end>']
pad_token = word_to_index['<pad>']
print(start_token, end_token, pad_token)
K = 1


# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# the associated function.
@app.route('/')
# ‘/’ URL is bound with hello_world() function.
def hello_world():


    return jsonify({'status': 'Server 2 is UP ...'})


@app.route('/foo', methods=['POST'])
def foo():
    data = request.json
    image_data_torch = torch.tensor(data['image_embedding'])
    print(image_data_torch.shape)

    img_embed = image_data_torch.permute(0, 2, 3, 1)
    img_embed = img_embed.view(img_embed.size(0), -1, img_embed.size(3))

    input_seq = [pad_token] * max_seq_len
    input_seq[0] = start_token

    input_seq = torch.tensor(input_seq).unsqueeze(0)
    predicted_sentence = []
    # return {'tt':"ok"}
    with torch.no_grad():
        for eval_iter in range(0, max_seq_len):

            output, padding_mask = model.forward(img_embed, input_seq)

            output = output[eval_iter, 0, :]

            values = torch.topk(output, K).values.tolist()
            indices = torch.topk(output, K).indices.tolist()

            next_word_index = random.choices(indices, values, k=1)[0]

            next_word = index_to_word[next_word_index]

            input_seq[:, eval_iter + 1] = next_word_index

            if next_word == '<end>':
                break

            predicted_sentence.append(next_word)
    print("\n")
    print("Predicted caption : ")
    predicted_sentence[0]=predicted_sentence[0][0].upper()+predicted_sentence[0][1:]
    sentence = " ".join(predicted_sentence + ['.'])
    print(sentence)


    return {'prediction': f'{sentence}'}


# main driver function
if __name__ == '__main__':
    # run() method of Flask class runs the application
    # on the local development server.
    app.run(port=5001,use_reloader=False)



