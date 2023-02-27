from kobert import get_pytorch_kobert_model
from kobert import get_tokenizer
from gluonnlp.data import SentencepieceTokenizer
from dataset import MuLTdataset
from getlist import get_list
from model import BPM_ST, BPM_MT
import torch
import torch.nn.functional as F

is_MT = False   # True for MT, False for ST
use_CUDA = True # True for GPU, False for CPU
batch_size = 64 # Batch size
epochs = 60     # Number of epochs

def main():
    _list=get_list()
    annotationdir='/data/rlaalsrl0922/BC/BCprediction/etri.tsv'
    csAUDIO_DIR = "1. 상담자 녹음본.wav"
    clAUDIO_DIR = "2. 내담자 녹음본.wav"
    device=torch.device("cuda")
    tokenizer = SentencepieceTokenizer(get_tokenizer())
    bert, vocab = get_pytorch_kobert_model()
    f1_score = 0
    

    dataset=MuLTdataset(annotationdir,csAUDIO_DIR,clAUDIO_DIR,tokenizer,vocab,_list)
    train_dataset = torch.utils.data.Subset(dataset, range(int(len(dataset)*0.8))) # Dataset not implemented yet
    val_dataset =torch.utils.data.Subset(dataset, range(int(len(dataset)*0.8),len(dataset)))  # Dataset not implemented yet

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    if is_MT:
        model = BPM_MT(tokenizer=tokenizer, bert=bert, vocab=vocab)
    else:
        model = BPM_ST(bert=bert).to(device)
    
    # Get the model parameters divided into two groups : bert and others
    bert_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'bert' in name:
            bert_params.append(param)
        else:
            other_params.append(param)
            

    adam_optimizer = torch.optim.Adam(bert_params, lr=0.0005)
    sgd_optimizer = torch.optim.SGD(other_params, lr=0.0005)


    # Training loop
    for epoch in range(epochs):
        for batch in train_dataloader:
            signal,sentence,label = batch
            print(label.shape)
            sentence=torch.squeeze(sentence,1)
            if use_CUDA:
                signal = signal.to(device)
                sentence=sentence.to(device)
            # Move the batch to GPU if CUDA is available
            y = model(signal,sentence)

            # Get the logit from the model
            logit     = y["logit"]
            label=label.to('cpu')
            logit=logit.to('cpu')
            print(f"logit shape is :{logit.shape}, label shape is : {label.shape}") #64,8 #64,
            # Calculate the loss
            loss_BC = F.cross_entropy(logit,  label)
            if is_MT:
                sentiment = y["sentiment"]
                loss_SP = F.binary_cross_entropy(torch.sigmoid(sentiment), sentiment)
                loss = 0.9 * loss_BC +  0.1 * loss_SP
            else:
                loss = loss_BC

            # Backpropagation
            loss.backward()

            # Update the model parameters
            adam_optimizer.step()
            sgd_optimizer.step()

            # Zero the gradients
            adam_optimizer.zero_grad()
            sgd_optimizer.zero_grad()

            print("Epoch : {}, Loss : {}".format(epoch, loss.item()))

        # Validation loop
        accuracy = 0
        loss     = 0
        
        for batch in val_dataloader:
            signal,sentence,label = batch
            print(sentence.shape)
            sentence=torch.squeeze(sentence,1)
            if use_CUDA:
                signal = signal.to(device)
                sentence=sentence.to(device)

            # Get the logit from the model
            logit     = y["logit"]
            label=label.to('cpu')
            logit=logit.to('cpu')

            # Calculate the loss
            
            loss_BC = F.cross_entropy(logit,label)
            if is_MT:
                sentiment = y["sentiment"]
                loss_SP = F.binary_cross_entropy(torch.sigmoid(sentiment), sentiment)
                loss = 0.9 * loss_BC +  0.1 * loss_SP
            else:
                loss = loss_BC

            # Calculate the accuracy
            accuracy += (torch.argmax(logit, dim=1) == label).sum().item()
            loss    += loss.item()

        accuracy /= len(val_dataset)
        loss     /= len(val_dataloader)
        print("Epoch : {}, Accuracy : {}, Loss : {}".format(epoch, accuracy, loss))

if __name__ == "__main__":
    main()