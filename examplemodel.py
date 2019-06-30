"""Example Model format"""

from model import Model

class exampleModel(Model):

    def __init__(self):
        """Initialize Example Model"""
        self.classvar = None

    def learn(self, input):
        """Add method for learning here"""
        #return output
        return NotImplementedError

    def predict(self, input):
        """Add method for predicting here"""
        #return output
        return NotImplementedError
