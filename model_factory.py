################################################################################
# CSE 151B: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Updated by Rohin
# Winter 2022
################################################################################
# Imports
#from base64 import encode
#from multiprocessing.reduction import duplicate
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
# Build and return the model here based on the configuration.
class Encoder(nn.Module):
    """
    Encoder CNN -> resnet50
    """
    def __init__(self, embed_dim, use_pretrained=True):
        super(Encoder, self).__init__()
        # Use pretrained resnet
        self.model = models.resnet50(pretrained=use_pretrained)
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        # Extract and replace only last trainable fc layer
        num_ftrs = self.model.fc.in_features
        self.fc = nn.Linear(num_ftrs, embed_dim)
        self.model.fc = self.fc

    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):
    """
    LSTM/RNN Decoder for Architectures 1 and 2
    """
    def __init__(self, embed_dim, num_embeds, hidden_size, cell_type, architecture, num_layers=2):
        super(Decoder, self).__init__()
        self.cell_type = cell_type
        self.architecture = architecture
        self.word_embedding = nn.Embedding(num_embeds, embed_dim)
        if architecture == 2:
            self.lstm = nn.LSTM(embed_dim*2, hidden_size, num_layers, batch_first=True)
        else: self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers, batch_first=True)
        self.rnn = nn.RNN(embed_dim, hidden_size, num_layers, nonlinearity="relu", batch_first=True)
        self.fc = nn.Linear(hidden_size, num_embeds)

    def forward(self, features, captions):
        if self.architecture == 1:
            return self.arch1_forward(features, captions)
        elif self.architecture == 2:
            return self.arch2_forward(features, captions)
        else:
            raise TypeError("Architecture number is invalid")

    def arch1_forward(self, features, captions):
        word_embeds = self.word_embedding(captions[:, :-1])
        input_seq = torch.cat((features.unsqueeze(dim=1), word_embeds), dim=1)
        if self.cell_type == "LSTM":
            hidden_states, _  = self.lstm(input_seq)
        elif self.cell_type == "RNN":
            hidden_states, _  = self.rnn(input_seq)
        else:
            raise TypeError("Cell type is invalid")
        return self.fc(hidden_states)

    def arch2_forward(self, features, captions):
        # TODO !!!
        captions = captions[:, :-1]
        pad = torch.zeros([captions.shape[0], 1], dtype=torch.long).cuda()
        padded_caps = torch.cat((pad, captions), dim=1)
        embeds = self.word_embedding(padded_caps)
        duplicate = features.unsqueeze(1).expand(embeds.size())
        input_seq = torch.cat((duplicate, embeds), dim=2)
        hidden_states, _  = self.lstm(input_seq)
        return self.fc(hidden_states)

class Baseline(nn.Module):
    def __init__(self, embed_dim, num_embeds, hidden_size, vocab, cell_type="LSTM", architecture=1):
        super(Baseline, self).__init__()
        self.embed_dim = embed_dim
        self.num_embeds = num_embeds
        self.hidden_size = hidden_size
        self.cell_type = cell_type
        self.architecture = architecture
        self.vocab = vocab
        # Initialize encoder/decoder
        self.encoder = Encoder(self.embed_dim)
        self.decoder = Decoder(self.embed_dim, self.num_embeds, self.hidden_size, self.cell_type, self.architecture)
    
    def forward(self, images, captions):
        features = self.encoder(images)
        embeds = self.decoder(features, captions)
        return embeds
    
    def generate_captions(self, images, max_length, deterministic, temperature):
        out_captions = np.zeros((images.shape[0], max_length))
        features = self.encoder(images).unsqueeze(1)
        input_seq = features
        states = None; hidden_states = None
        for i in range(max_length):
            hidden_states, states = self.decoder.lstm(input_seq,states)
            output = self.decoder.fc(hidden_states).squeeze(1)
            if deterministic: 
                distr = output.argmax(axis=1).unsqueeze(1)
            else: 
                probs = F.softmax(output.div(temperature))
                distr = torch.multinomial(probs, 1)
            out_captions[:,i] = distr.cpu().numpy().squeeze(1)
            input_seq = self.decoder.word_embedding(distr.cuda())
        final_captions = []
        for caption in out_captions:
            caption = caption.tolist()
            trimmed = caption[:caption.index(0)] if 0 in caption else caption #np.where(caption==0)[0] if np.where(caption==0)[0] else len(caption)
            trimmed_words = [self.vocab.idx2word[int(x)] for x in trimmed if x > 3]
            final_captions.append(list(trimmed_words))

        return final_captions

class VanillaRNN(nn.Module):
    def __init__(self, embed_dim, num_embeds, hidden_size, vocab, cell_type="RNN", architecture=1):
        super(VanillaRNN, self).__init__()
        self.embed_dim = embed_dim
        self.num_embeds = num_embeds
        self.hidden_size = hidden_size
        self.cell_type = cell_type
        self.architecture = architecture
        self.vocab = vocab
        # Initialize encoder/decoder
        self.encoder = Encoder(self.embed_dim)
        self.decoder = Decoder(self.embed_dim, self.num_embeds, self.hidden_size, self.cell_type, self.architecture)
    
    def forward(self, images, captions):
        features = self.encoder(images)
        embeds = self.decoder(features, captions)
        return embeds
    
    def generate_captions(self, images, max_length, deterministic, temperature):
        out_captions = np.zeros((images.shape[0], max_length))
        features = self.encoder(images).unsqueeze(1)
        input_seq = features
        states = None; hidden_states = None
        for i in range(max_length):
            hidden_states, states = self.decoder.rnn(input_seq,states)
            output = self.decoder.fc(hidden_states).squeeze(1)
            if deterministic: 
                distr = output.argmax(axis=1).unsqueeze(1)
            else: 
                probs = F.softmax(output.div(temperature))
                distr = torch.multinomial(probs, 1)
            out_captions[:,i] = distr.cpu().numpy().squeeze(1)
            input_seq = self.decoder.word_embedding(distr.cuda())
        final_captions = []
        for caption in out_captions:
            caption = caption.tolist()
            trimmed = caption[:caption.index(0)] if 0 in caption else caption
            trimmed_words = [self.vocab.idx2word[int(x)] for x in trimmed if x > 3]
            final_captions.append(list(trimmed_words))

        return final_captions

class BaselineArch2(nn.Module):
    def __init__(self, embed_dim, num_embeds, hidden_size, vocab, cell_type="LSTM", architecture=2):
        super(BaselineArch2, self).__init__()
        self.embed_dim = embed_dim
        self.num_embeds = num_embeds
        self.hidden_size = hidden_size
        self.cell_type = cell_type
        self.architecture = architecture
        self.vocab = vocab
        # Initialize encoder/decoder
        self.encoder = Encoder(self.embed_dim)
        self.decoder = Decoder(self.embed_dim, self.num_embeds, self.hidden_size, self.cell_type, self.architecture)
    
    def forward(self, images, captions):
        features = self.encoder(images)
        embeds = self.decoder(features, captions)
        return embeds
    
    def generate_captions(self, images, max_length, deterministic, temperature):
        out_captions = np.zeros((images.shape[0], max_length))
        features = self.encoder(images).unsqueeze(1)
        input_seq = torch.cat((features, self.decoder.word_embedding(torch.zeros((images.shape[0], 1)).long().cuda())), dim=2)
        states = None; hidden_states = None
        for i in range(max_length):
            hidden_states, states = self.decoder.lstm(input_seq,states)
            output = self.decoder.fc(hidden_states).squeeze(1)
            if deterministic: 
                distr = output.argmax(axis=1).unsqueeze(1)
            else: 
                probs = F.softmax(output.div(temperature))
                distr = torch.multinomial(probs, 1)
            out_captions[:,i] = distr.cpu().numpy().squeeze(1)
            input_seq = torch.cat((features, self.decoder.word_embedding(distr.cuda())), dim=2)
        final_captions = []
        for caption in out_captions:
            caption = caption.tolist()
            trimmed = caption[:caption.index(0)] if 0 in caption else caption #np.where(caption==0)[0] if np.where(caption==0)[0] else len(caption)
            trimmed_words = [self.vocab.idx2word[int(x)] for x in trimmed if x > 3]
            final_captions.append(list(trimmed_words))
        return final_captions


def get_model(config_data, vocab):
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type']
    vocab_size = vocab.__len__()
    print(model_type)
    model = None
    if model_type == "baseline":
        model = Baseline(embedding_size, vocab_size, hidden_size, vocab)
    elif model_type == "vanilla_rnn":
        model = VanillaRNN(embedding_size, vocab_size, hidden_size, vocab)
    elif model_type == "arch2":
        model = BaselineArch2(embedding_size, vocab_size, hidden_size, vocab)

    if model is not None:
        return model
    else:
        raise TypeError("Model is NoneType")
