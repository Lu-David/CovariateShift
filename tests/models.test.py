import unittest

from rba.models.rba_classifier import RBAClassifier
import torch  
import os
import numpy as np
import scipy

class RBATestCase(unittest.TestCase):   

    def setUp(self) -> None:
        self.r_st = torch.rand((100, 1))
        self.x = torch.rand((100, 2))
        self.y = torch.round(torch.rand((100, 1)))
        self.model = RBAClassifier()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.optimizer.zero_grad()

    def testRSTDoesNotChange(self):
        temp = self.r_st.clone()
        outputs = self.model(self.x, self.r_st)
        outputs.backward(self.y)
        self.optimizer.step()

        self.assertTrue(torch.equal(temp, self.r_st))
        

if __name__ == '__main__':
    unittest.main()