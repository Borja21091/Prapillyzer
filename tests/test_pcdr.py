import unittest
import numpy as np
from superfeatures import cdr_profile

class TestCdrProfile(unittest.TestCase):

    def test_invalid_mask_dimensions(self):
        mask = np.zeros((100, 100, 3), dtype=np.uint8)
        with self.assertRaises(AssertionError):
            cdr_profile(mask)

    def test_invalid_mask_dtype(self):
        mask = np.zeros((100, 100), dtype=np.float32)
        with self.assertRaises(AssertionError):
            cdr_profile(mask)

    def test_invalid_mask_pixel_values(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[50, 50] = 1
        with self.assertRaises(AssertionError):
            cdr_profile(mask)

    def test_invalid_angular_step(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[50:60, 50:60] = 1
        mask[70:80, 70:80] = 2
        with self.assertRaises(AssertionError):
            cdr_profile(mask, ang_step=7)

if __name__ == '__main__':
    unittest.main()