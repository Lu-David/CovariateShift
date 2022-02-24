import unittest

from rba.train.rba_train import rba_train
import torch  
import os
import numpy as np
import scipy

class TrainRBATestCase(unittest.TestCase):   

    def setUp(self) -> None:
        self.r_st = torch.rand((100, 1))
        self.r_ts = torch.ones((100, 1))

        self.x = torch.rand((100, 2))
        self.y = torch.round(torch.rand((100, 1)))

        rba_train(self.x, self.y, self.r_st, self.r_ts)
        

    def testRSTDoesNotChange(self):
        temp = self.r_st.clone()
        outputs = self.model(self.x, self.r_st)
        outputs.backward(self.y)
        self.optimizer.step()

        self.assertTrue(torch.equal(temp, self.r_st))
        

if __name__ == '__main__':
    unittest.main()