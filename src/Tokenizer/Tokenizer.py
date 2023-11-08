from tokenizers import Tokenizer
import numpy as np
from typing import Iterable, Union

class NLPTokenizer:
    """
    NLPTokenizer is used as a pipeline to encode and decode a text to vector and vector to text
    """
    def __init__(self, tokenizerFile:str, sequenceLength:int) -> None:
        """
        NLPTokenizer is used as a pipeline to encode and decode a text to vector and vector to text
        
        Keyword arguments:

        tokenizerFile (str) -- Path to the tokenizer json file

        sequenceLength (int) -- Maximum tokens

        Return: None
        """
        
        self.tokenizer = Tokenizer.from_file(tokenizerFile)
        self.sequenceLength = sequenceLength

    def pad(self, tokens:Union[Iterable[int], np.ndarray]) -> np.ndarray:
        """
        Adds padding to vector using -1 for empty string, so it has the same length of `sequenceLength` of the object
        
        Keyword arguments:

        tokens (Union[Iterable[int], np.ndarray]) -- Tokens to be padded

        Return: A padded array
        """
        totalTokens = len(tokens)

        if totalTokens > self.sequenceLength:
            raise ValueError(f"The given text should contain less than {self.sequenceLength} not {totalTokens}")
        
        if totalTokens < self.sequenceLength:
            tokens = np.pad(tokens, (self.sequenceLength-totalTokens, 0), 'constant', constant_values=0)

        return tokens
    
    def encode(self, text:str, shouldPad=True) -> Union[np.ndarray, list[list[int]]]:
        """
        Tokenizes a text and return a matrix of shape, (n, sequenceLength)

        Here `n` is dependant on the length of text
        `n = len(tokenizer.encode(text).ids) % sequenceLength`
        
        Keyword arguments:

        text (str) -- Text to be padded

        shouldPad (bool) -- Will pad the tokens and if true

        Return: A padded array
        """

        tokens = self.tokenizer.encode(text).ids

        tokens = [tokens[i:i+self.sequenceLength] for i in range(0, len(tokens), self.sequenceLength)]

        if shouldPad:
            return np.vstack( list( map( lambda x: self.pad(x), tokens ) ))
        else:
            return np.array(tokens) if len(tokens) == 1 else tokens
        
    def decode(self, tokens:Union[np.ndarray, list[list[int]]]) -> list[str]:
        """
        Decodes an encoded matrix
        
        Keyword arguments:

        Tokens -- Tokens in the form of a numpy array or list of lists

        Return: Decoded text
        """

        unpaddedTokens = map(lambda x: x[x != 0], tokens)

        return list(map(lambda x: self.tokenizer.decode(x).replace(" ##", ""), unpaddedTokens))   