import numpy as np
from DataLoader.Errors import PrmotionError
from typing import Union

class Paginator:
    """
    # Paginator

    Paginator holds tensor of given shape let's say `(m, n)`. Here `m` is the batchsize and `n` is the size of the vector. It acts as an iterator, only returning matrix of height `n` and storing remaining data to be accessed later.

    ## How it works?

    This class will be used as a parent for other classes, and each child should  have a `nextBatch` method, `__next__` method wil use the `nextBatch` method to get data and store the later data. If the nextBatch returns data with size less than `m`, then the `nextBatch` method will be called again until batch size >= `m`
    """
    def __init__(self, batchSize:int, shouldShuffle=True) -> None:
        """
        Initiates the object
        
        Keyword arguments:

        batchSize -- number of items in the batch

        shouldShuffle -- shuffles data if true

        Return: Initiates the object
        """
        self.batchSize = batchSize
        # setting the shape 

        self.cache1:np.ndarray = np.ndarray((0))
        # stores data of maximum size batchSize 
        self.cache2:np.ndarray = np.ndarray((0))
        # can store data of any size 

        # initiating cache 

        self.shouldShuffle = shouldShuffle

    def nextBatch(self) -> np.ndarray:
        """
        This method should be implemented by the child and return a numpy array
        """
        raise NotImplementedError("This method should be implemented by the child and return a numpy array")
    

    def promoteData(self, a:np.ndarray, b:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Promotes data from one level to higher
        
        Keyword arguments:

        a -- Numpy array in which data is promoted `to` 

        b -- Numpy array in which data is promoted `from` 

        Return: Returns updated `a` and `b`
        """
        
        aSize = a.shape[0]  
        bSize = b.shape[0]

        if bSize == 0 and aSize<=self.batchSize:
            raise PrmotionError("Nothing left to promote")
        
        if aSize==self.batchSize:
            return a, b

        if aSize < self.batchSize and bSize > 0:
            removeChunk = self.batchSize-aSize if (bSize > self.batchSize-aSize) else bSize
            updateData = b[:removeChunk]
            # getting new data to be pushed to a 

            b = b[removeChunk:]
            # updating b

            if a.shape[0] > 0:
                a = np.vstack((updateData, a))
            else:
                a = updateData

        elif aSize > self.batchSize:
            removeChunk = aSize-self.batchSize-1
            updateData = a[removeChunk:]
            # getting data to be deprmotoed 

            a = a[:self.batchSize]
            # updating a 

            if b.shape[0] > 0:
                b = np.vstack((b, updateData))
            else:
                b = updateData


        
        return a, b
    
    def updateCache(self, data:np.ndarray) -> np.ndarray:
        """
        Updates the current cache
        
        Keyword arguments:

        data -- Data for the cache to be updated on

        Return: Updated data
        """
        try:
            self.cache1, self.cache2 = self.promoteData(self.cache1, self.cache2)    
        except PrmotionError:
            pass
        try:
            updatedData, self.cache1 = self.promoteData(data, self.cache1)
            return updatedData
        except PrmotionError:
            pass

        return data
        
    def __iter__(self):
        return self
    
    def retrieveNextBatch(self) -> np.ndarray:
        """
        Calls nextBatch, returns the data returned, if it is bigger than batchsize, if not, calls itself again
        
        Keyword arguments:

        currentData -- Current data

        Return: next batch
        """
        currentData = self.nextBatch()

        while currentData.shape[0] <= self.batchSize:
            currentData = np.vstack((currentData, self.nextBatch()))

        return currentData


    def __next__(self):
        """
        Returns the next data
        """
        if self.cache1.shape[0] < self.batchSize:
            futureData = self.retrieveNextBatch()
            newData = self.updateCache(futureData)
        else:
            newData = self.updateCache(np.ndarray((0, *self.cache1.shape[1:]), self.cache1.dtype))

        if self.shouldShuffle:
            shuffleIndices = np.random.permutation(newData.shape[0])
            newData = newData[shuffleIndices]

        return newData



        