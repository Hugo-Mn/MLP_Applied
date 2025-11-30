from transformers import RemBertTokenizer, RemBertModel, Wav2Vec2Processor, Wav2Vec2Model, AutoFeatureExtractor
import torch as t
import torchaudio as ta
import librosa
import numpy as np

class Encoder:
    def __init__(self):
        self.device = t.device("cuda" if t.cuda.is_available() else "cpu")
        print(f"[Encoder] Using device: {self.device}")

        self.bert = RemBertModel.from_pretrained("google/rembert").to(self.device)
        self.txtTokenizer = RemBertTokenizer.from_pretrained("google/rembert")
        self.xlsrProcessor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-300m")
        self.xlsr = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m").to(self.device)
        self.bert.eval()
        self.xlsr.eval()

    def convertOpusToWav(self, audioPath):
        audio, sr = librosa.load(audioPath, sr=16000, mono=True)
        audio = audio.astype("float32")
        return audio

    def readFile(self, txtPath):
        with open(txtPath, "r", encoding="utf-8") as file:
            lines = file.readlines()
        target = lines[0].strip() if len(lines) > 0 else None
        text = lines[1].strip() if len(lines) > 1 else ""
        return text, target

    #bert embedding 1152
    def encodeText(self, txtPath=None):
        if txtPath is None:
            return t.zeros((1, 2304)).to(self.device), None
        text, target = self.readFile(txtPath)
        if not text or text == "":
            return t.zeros((1, 2304)).to(self.device), target
        inputs = self.txtTokenizer(text, return_tensors="pt", padding=True, truncation=True)
        # Move inputs to GPU
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with t.no_grad():
            outputs = self.bert(**inputs)
        embeddingsMax = outputs.last_hidden_state.max(dim=1).values
        embeddingsMean = outputs.last_hidden_state.mean(dim=1)
        return t.cat((embeddingsMax, embeddingsMean), dim=1), target

    #xls embdding 1024
    def encodeAudio(self, audioPath=None):
        if audioPath is None:
            return t.zeros((1, 2048)).to(self.device)
        audio = self.convertOpusToWav(audioPath)
        inputs = self.xlsrProcessor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        # Move inputs to GPU
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with t.no_grad():
            outputs = self.xlsr(**inputs)
        embeddingsMax = outputs.last_hidden_state.max(dim=1).values
        embeddingsMean = outputs.last_hidden_state.mean(dim=1)
        return t.cat((embeddingsMax, embeddingsMean), dim=1)

    def encodeOneSet(self, txtPath, audioPath):
        textEmbedding, target = self.encodeText(txtPath)
        audioEmbedding = self.encodeAudio(audioPath)
        finalInput = t.cat((textEmbedding, audioEmbedding), dim=1)
        return finalInput, target

    def createTarget(self, country):
        match(country):
            case "French": return [1,0,0,0,0]
            case "Portugese": return [0,1,0,0,0]
            case "Italia": return [0,0,1,0,0]
            case "Spain": return [0,0,0,1,0]
            case "Polish": return [0,0,0,0,1]

