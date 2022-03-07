import unittest

from rba.models.rba_classifier import RBAClassifierSimple
import torch  
import torchviz
import os
import numpy as np
import scipy

class RBATestCase(unittest.TestCase):   

    def setUp(self) -> None:
        self.r_st = torch.rand((100, 1))
        self.r_ts = torch.ones((100, 1))

        self.x = torch.rand((100, 2))
        self.x *= self.r_ts

        self.y = torch.round(torch.rand((100, 1)))

        self.model = RBAClassifierSimple()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.optimizer.zero_grad()

    def testInputDoesNotChange(self):
        temp_x = self.x.clone()
        temp_rst = self.r_st.clone()
        temp_rts = self.r_ts.clone()

        outputs = self.model(self.x, self.r_st)
        print(torchviz.make_dot(outputs, params=dict(self.model.named_parameters())))
        outputs.backward(self.y)

        self.optimizer.step()

        self.assertTrue(torch.equal(temp_rst, self.r_st))
        self.assertTrue(torch.equal(temp_rts, self.r_ts))
        self.assertTrue(torch.equal(temp_x, self.x))

    def testModelPolyFeatures(self):
        self.model = RBAClassifierSimple(poly_features=2)
        temp_x = self.x.clone()
        temp_rst = self.r_st.clone()
        temp_rts = self.r_ts.clone()

        outputs = self.model(self.x, self.r_st)
        outputs.backward(self.y)

        self.optimizer.step()

        self.assertTrue(torch.equal(temp_rst, self.r_st))
        self.assertTrue(torch.equal(temp_rts, self.r_ts))
        self.assertTrue(torch.equal(temp_x, self.x))


if __name__ == '__main__':
    unittest.main()