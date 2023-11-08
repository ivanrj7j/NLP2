import numpy as np
from DataLoader import Paginator
from Tokenizer import NLPTokenizer
from pandas import read_csv
from tensorflow.data import Dataset
from tensorflow import TensorSpec, constant, one_hot  
from tensorflow import uint8 as tfUInt8
from tensorflow import uint16 as tfUInt16

class TrainLoader(Paginator):
    """
    # Train Loader

    Goes through each text file and converts them to tokens, if the tokens are over sequencelength, it gives more than one batch
    """
    
    def __init__(self, batchSize: int, sequenceLength:int, tokenizerFile:str, dataFile:str, shouldShuffle=True, dataFileChunkSize:int=32) -> None:
        """
        Initiates module
        
        Keyword arguments:

        batchSize -- number of items in the batch

        sequenceLength -- Maximum tokens

        tokenizerFile -- Path to the tokenizer json file

        dataFile -- Path to the csv data file

        shouldShuffle -- shuffles data if true

        dataFileChunkSize -- Chunk size of the data file

        Return: None
        """

        super().__init__(batchSize, shouldShuffle)
        self.sequenceLength = sequenceLength
        self.tokenizer = NLPTokenizer(tokenizerFile, sequenceLength+1) # added 1 to have one reserved as label
        # setting up tokenizer 

        self.dataFile = dataFile
        self.dataFileChunkSize = dataFileChunkSize

        self.regenerateData()
        # setting training data 

    def regenerateData(self):
        self.data = read_csv(self.dataFile, index_col=False, chunksize=self.dataFileChunkSize)

    def nextBatch(self) -> np.ndarray:
        """
        This method returns the next batch using the loaded data
        """
        try:
            data = self.data.__next__()['0']
            x, y = zip(*data.apply(lambda j: self.trainTokenizeText(j)))

            return np.hstack((np.vstack(x), np.hstack(y).reshape(-1, 1)))
        except StopIteration:
            self.regenerateData()
            raise StopIteration()
        
    
    def trainTokenizeText(self, text:str) -> np.ndarray:
        """
        Tokenizes a text for passing through training 
        
        Keyword arguments:

        text -- text to be tokenized

        Return: a matrix of tokens
        """
        encodedMatrix = self.tokenizer.encode(text)
        return encodedMatrix[:, :-1], encodedMatrix[:, -1]
    
    def __next__(self):
        array = super().__next__()

        return np.hsplit(array, [-1])   

    def getTensorflowDataset(self) -> Dataset:
        """
        Returns a tensorflow dataset based on the loader
        """

        def generatorFunc():
            for x, y in self:
                yield constant(x, tfUInt16), one_hot(y.reshape((-1)), self.tokenizer.tokenizer.get_vocab_size(), dtype=tfUInt8)
        
        return Dataset.from_generator(generatorFunc, output_signature=((
            TensorSpec(shape=(self.batchSize, self.sequenceLength), dtype=tfUInt16),  # Features
            TensorSpec(shape=(self.batchSize, self.tokenizer.tokenizer.get_vocab_size()), dtype=tfUInt8) # Labels
        )))

       