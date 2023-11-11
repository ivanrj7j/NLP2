import numpy as np
from DataLoader import Paginator
from Tokenizer import NLPTokenizer
from tensorflow.data import Dataset
from tensorflow import TensorSpec, constant, one_hot  
from tensorflow import uint16 as tfInt16
from tensorflow import uint8 as tfInt8
from pandas import read_csv
from copy import deepcopy

class TrainLoader(Paginator):
    """
    # TrainLoader

    TrainLoader is used for loading data for training NLP models, this module is a child of `Paginator`

    TrainLoader reads through a pandas iterable and tokenizes text using `NLPTokenizer`
    """
    
    def __init__(self, batchSize: int, sequenceLength:int, tokenizerFile:str, dataFile:str, dataFileChunkSize:int, shouldShuffle=True) -> None:
        """
        Initiates module
        
        Keyword arguments:

        batchSize -- number of items in the batch

        sequenceLength (int) -- Maximum tokens

        tokenizerFile (str) -- Path to the tokenizer json file

        dataFile (str) -- Path to the csv file containg data

        dataFileChunkSize (str) -- Size of 1 chunk to load from csv

        shouldShuffle -- shuffles data if true

        Return: None
        """
        
        super().__init__(batchSize, shouldShuffle)
        self.tokenizer = NLPTokenizer(tokenizerFile, sequenceLength)
        self.sequenceLength = sequenceLength
        
        self.dataFile = dataFile
        self.dataFileChunkSize = dataFileChunkSize

        self.data = read_csv(dataFile, chunksize=dataFileChunkSize, index_col=False)

    def nextBatch(self) -> np.ndarray:
        """
        This method returns the next batch using the loaded data
        """
        try:
            data = self.data.__next__()['0']
            x, y = zip(*data.apply(lambda x: self.trainTokenizeText(x)))

            return np.hstack((np.vstack(x), np.hstack(y).reshape(-1, 1)))
        except StopIteration:
            self.data = read_csv(self.dataFile, chunksize=self.dataFileChunkSize, index_col=False)

            raise StopIteration()
        
    
    def trainTokenizeText(self, text:str) -> np.ndarray:
        """
        Tokenizes a text for passing through training 
        
        Keyword arguments:

        text -- text to be tokenized

        Return: a matrix of tokens
        """
        
        encodedMatrix = self.tokenizer.encode(text, False)
        x = []
        y = []
        for vector in encodedMatrix:
            for i in range(1, len(vector)):
                start = max(0, i-self.sequenceLength)
                data = vector[start:i]
                if len(data) < self.sequenceLength:
                    data = np.pad(data, (self.sequenceLength-len(data), 0), 'constant', constant_values=0)
                x.append(data)
                y.append(vector[i])
        
        return np.vstack(x).astype(np.uint16), np.array(y, dtype=np.uint16)
    
    def __next__(self):
        array = super().__next__()

        return np.hsplit(array, [-1])   

    def getTensorflowDataset(self) -> Dataset:
        """
        Returns a tensorflow dataset based on the loader
        """

        def generatorFunc():
            for x, y in self:
                yield constant(x, tfInt16), one_hot(y.reshape((-1)), self.tokenizer.tokenizer.get_vocab_size(), dtype=tfInt8)
        
        return Dataset.from_generator(generatorFunc, output_signature=((
            TensorSpec(shape=(self.batchSize, self.sequenceLength), dtype=tfInt16),  # Features
            TensorSpec(shape=(self.batchSize, self.tokenizer.tokenizer.get_vocab_size()), dtype=tfInt8) # Labels
        )))