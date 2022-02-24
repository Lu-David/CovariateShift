import unittest

import rba.density_estimation as de
import torch  

class OnesTestCase(unittest.TestCase):   

    def setUp(self) -> None:
        self.x = torch.rand((100, 5))
        self.ones = de.ones

    def testOnesReturnsCorrectTensor(self):
        ones = self.ones(self.x)
        self.assertEqual(type(ones), torch.Tensor)
        self.assertTrue(ones.shape, torch.Size([100, 1]))    

    def testOnesName(self):
        self.assertTrue(self.ones.__name__ == "ones")

class KDETestCase(unittest.TestCase):   

    def setUp(self) -> None:
        self.X_s = torch.rand((100, 5))
        self.X_t = torch.rand((100, 5))
        self.kde = de.get_kernel_density_estimator(self.X_s, self.X_t)

    def testKDEReturns1DTensor(self):
        tensor = self.kde(self.X_s)
        self.assertEqual(type(tensor), torch.Tensor)
        self.assertEqual(tensor.shape, torch.Size([100, 1]))

    def testKDEName(self):
        self.assertTrue(self.kde.__name__ == "kde")

class GMMTestCase(unittest.TestCase):   

    def setUp(self) -> None:
        self.X_s = torch.rand((100, 2))
        self.X_t = torch.rand((100, 2))
        self.gmm = de.get_gmm_estimator(self.X_s, self.X_t)

    def testGMMReturns1DTensor(self):
        tensor = self.gmm(self.X_s)
        self.assertEqual(type(tensor), torch.Tensor)
        self.assertEqual(tensor.shape, torch.Size([100, 1]))

    def testGMMName(self):
        self.assertTrue(self.gmm.__name__ == "gmm")

# class LRDRTestCase(unittest.TestCase):   

#     def setUp(self) -> None:
#         self.X_s = torch.rand((100, 5))
#         self.X_t = torch.rand((100, 5))
#         self.gmm = de.get_lrdr_estimator(self.X_s, self.X_t)

#     def testLRDRReturns1DTensor(self):
#         tensor = self.gmm(self.X_s)
#         self.assertEqual(type(tensor), torch.Tensor)
#         self.assertEqual(tensor.shape, torch.Size([100, 1]))

#     def testLRDRName(self):
#         self.assertTrue(self.gmm.__name__ == "lrdr")
 
if __name__ == '__main__':
    unittest.main()