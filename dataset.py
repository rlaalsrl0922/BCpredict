from torch.utils.data import Dataset
from pydub import AudioSegment
import torchaudio
import torchaudio.transforms as T
import pandas as pd
import os
import torch

class MuLTdataset(Dataset):
    def __init__(self,annotations_file,csaudio,claudio,tokenizer,vocab,_list):
        self.annotations_file = self.annotations=pd.read_csv(annotations_file,delimiter='\t',encoding='utf-8')
        self.csaudio=csaudio
        self.claudio=claudio
        self.targetsr=22050
        self.numsamples=33075
        self.length=32
        self.tokenizer=tokenizer
        self.vocab=vocab
        self._dict=_list


    def __len__(self):
        return len(self.annotations_file)

    def __getitem__(self, index):
        label = self._get_audio_sample_label(index)
        transcription = self._get_transcription(index)
        folder = self.get_folder(index)
        
        #extract audio feature
        audio_sample_path=self._get_audio_sample_path(index,folder)
        subdr = self._gettime(index,audio_sample_path)
        signal, sr = torchaudio.load(subdr,normalize=True)
        signal = self._resample(signal , sr)
        #signal = self._mixdown(signal)
        signal = self._cutout(signal)
        signal = self._padding(signal)
        signal = self._tomfcc(signal,subdr)
        
        #extract text feature
        sentences = self._cutword(transcription)
        sentences = self.transform(sentences)
        sentences = self.pad(sentences)
        
        return signal, sentences, label
    
    
    #setting
    def _get_transcription(self,index):
        return self.annotations.iloc[index,1]
    
    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 2]
    
    def get_folder(self,index):
        folder = self._dict[self.annotations.iloc[index,6]-1]
        return folder
    
    
    #audio to mfccs
    def _get_audio_sample_path(self,index,fold):
        if(self.annotations.iloc[index,5]=="counselor"):
            path=os.path.join('/data/datasets/ETRI_Backchannel_Corpus_2022/'+fold+'/',self.csaudio)
        else:
            path=os.path.join('/data/datasets/ETRI_Backchannel_Corpus_2022/'+fold+'/',self.claudio)
        return path
    
    
    #[Errno 2] No such file or directory: 'fold1/1. 상담자 녹음본.wav'
    def _gettime(self,index,dir):
        sound = AudioSegment.from_file(dir)
        start=self.annotations.iloc[index,3]
        end=self.annotations.iloc[index,4]
        StartTime=float(start)*1000
        Endtime=float(end)*1000
        extract=sound[StartTime:Endtime]
        extract.export(str(index)+".wav", format="wav")
        dr=str(index)+".wav"
        return dr
    
    def _resample(self, signal, sr):
        if(signal.shape[1]>0):
            if sr != self.targetsr:
                resampler = torchaudio.transforms.Resample(sr, self.targetsr)
                signal = resampler(signal)
        return signal

    #채널이 많을때 한개로 바꾸기 위함 사용할필요없음 기존의 채널이 1이라서
    def _mixdown(self, signal): # signal -> (# channels, samples), ex) (2,16000) -> (1,16000)
        if signal.shape[0] > 1: # 채널이 1이상일때만 하기
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal
    
    #signal -> tensor -> (1,num_samples)일때, num_samples는 초기에 우리가 정한 22050 보다 큼
    def _cutout(self, signal):
        if signal.shape[1] > self.numsamples:
            signal = signal[:, :self.numsamples]
        return signal
    
    #지정된 시간보다 짧을경우
    def _padding(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.numsamples:
            num_missing_samples = self.numsamples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal
    
    def _tomfcc(self,sig,dir):
        transformer = T.MFCC(sample_rate=22050,n_mfcc=13,melkwargs={'n_fft': 2048,'n_mels': 256,'hop_length': 512,'mel_scale': 'htk',})
        mfccs=transformer(sig)
        mfccs=mfccs.reshape(13,-1)
        os.remove(dir)
        return mfccs # 13,65
    
    #text
    def _cutword(self,transcription):
        sentence=transcription
        wordlist=sentence.split(" ")
        wordsplit= wordlist[-5:]
        _5wordsentece=" ".join(wordsplit)
        return _5wordsentece
    
    def transform(self,transcription):
        text=self.tokenizer(transcription) 
        text=torch.tensor(self.vocab(text)) # token,vocab -> 리스트 반환 , tensor -> [n]차원 반환
        text=text.reshape(1,-1)
        return text

    def pad(self,text):
      length_text = text.shape[1]
      if length_text < self.length:
            num_missing_samples = self.length - length_text
            last_dim_padding = (num_missing_samples, 0)
            text = torch.nn.functional.pad(text, last_dim_padding)
      #text=text.reshape(-1)
      return text
