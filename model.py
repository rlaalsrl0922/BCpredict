import torch
import torch.nn as nn

class BPM_ST(nn.Module):
    def __init__(self, bert=None, output_size=128, dropout=0.3):
        super(BPM_ST, self).__init__()

        self.bert = bert
        assert self.bert is not None, "bert and vocab must be provided"
        self.audio_feature_size=65
        self.dropout = nn.Dropout(dropout)
        self.LSTM = nn.LSTM(input_size=self.audio_feature_size, hidden_size=self.audio_feature_size, num_layers=4, batch_first=True, bidirectional=True)
        self.fc_layer = nn.Linear(768 + 26*self.audio_feature_size, output_size)
        self.fc_layer2 = nn.Linear(output_size, 8)

    def forward(self,x,z):
        
        audio = x
        text  = z
        B,N,C=x.size()
        batch_size=B
        _, text = self.bert(text) #text=64,1,768 
        #text=text.reshape(-1,32)
        y = {}
        
        # pass the MFCC feature to LSTM layer
        audio,(_,__) = self.LSTM(self.dropout(audio))

        text=text.reshape(B,-1)  #64,1,768 -> 64,768
        audio=audio.reshape(B,-1) #64,13,130 -> 64,1690
        x = torch.cat((audio, text), dim=1) # 64,2458

        x = self.fc_layer(self.dropout(x)) #64,128
        y["logit"] = self.fc_layer2(self.dropout(x)) # 64,8
        
        return y
    
class BPM_MT(nn.Module):
    def __init__(self, tokenizer=None, bert=None, vocab=None, mfcc_extractor=None, output_size=128, sentiment_output_size=64, dropout=0.3):
        super(BPM_ST, self).__init__()

        # get the bert model and tokenizer from arguments        
        # tokenizer = SentencepieceTokenizer(get_tokenizer())
        # bert, vocab = get_pytorch_kobert_model()
        self.bert = bert
        self.vocab = vocab
        self.tokenizer = tokenizer
        # if bert and vocab are not provided, raise an error
        assert self.bert is not None and self.vocab is not None, "bert and vocab must be provided"
        
        # define the MFCC extractor
        # self.mfcc_extractor = MFCC(sample_rate=sample_rate,n_mfcc=13)
        self.mfcc_extractor = mfcc_extractor
        self.audio_feature_size = mfcc_extractor.n_mfcc

        # define the LSTM layer, 4 of layers
        self.LSTM = nn.LSTM(input_size=self.audio_feature_size, hidden_size=self.audio_feature_size, num_layers=4, batch_first=True, bidirectional=True)

        self.dropout = nn.Dropout(dropout)
        # FC layer that has 128 of nodes which fed concatenated feature of audio and text
        self.fc_layer = nn.Linear(768 + self.audio_feature_size, output_size)
        self.sentiment_fc_layer = nn.Linear(768, sentiment_output_size)

    def forward(self, x,z):
        audio = x #[64,1,X]
        text  = z #[64,1,32]
        
        # tokenize the text if tokenizer is provided
        #if self.tokenizer is not None:
        #text = self.tokenizer(text)
        # convert the text to index
        #text = self.vocab(text)
        # extract the text feature from bert model
        _, text = self.bert(text)
        y = {}
        
        # extract the MFCC feature from audio
        audio = self.mfcc_extractor(audio)
        # reshape the MFCC feature to (batch_size, length, 13)
        audio = audio.permute(0, 2, 1)
        # pass the MFCC feature to LSTM layer
        audio, _ = self.LSTM(audio)
        print(audio,audio.shape)

        # concatenate the audio and text feature
        x = torch.cat((audio, text), dim=1)
        # pass the concatenated feature to FC layer
        y["logit"] = self.fc_layer(self.dropout(x))

        # pass the concatenated feature to sentiment FC layer
        y["sentiment"] = self.sentiment_fc_layer(self.dropout(text))
        
        return y