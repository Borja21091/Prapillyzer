import unittest
import numpy as np
from PIL import Image
from unittest.mock import patch
from main import level_image

class TestLevelImage(unittest.TestCase):

    @patch('superfeatures.mask_fovea')
    @patch('superfeatures.mask_disc')
    @patch('superfeatures.rotate_image')
    def test_level_image(self, mock_rotate_image, mock_mask_disc, mock_mask_fovea):
        # Create a dummy image
        img = Image.new('RGB', (512, 512), color = 'white')
        
        # Mock the masks
        mask_f = np.zeros((224, 224), dtype=np.uint8)
        mask_f[100, 100] = 1  # Dummy fovea centroid
        mask_d = np.zeros((512, 512), dtype=np.uint8)
        mask_d[300, 300] = 1  # Dummy disc centroid
        
        mock_mask_fovea.return_value = mask_f
        mock_mask_disc.return_value = mask_d
        
        # Mock the rotation
        rotated_img = np.array(img.rotate(45))  # Dummy rotated image
        mock_rotate_image.return_value = rotated_img
        
        # Call the function
        out_img, fov_coord, disc_coord, ang = level_image(img, mask_f, mask_d)
        
        # Assertions
        self.assertTrue(np.array_equal(out_img, rotated_img), "The image was not rotated correctly.")
        np.testing.assert_almost_equal(fov_coord, (100 * 512 / 224, 100 * 512 / 224), decimal=7, err_msg="Fovea centroid is incorrect.")
        self.assertEqual(disc_coord, (300, 300), "Disc centroid is incorrect.")
        self.assertAlmostEqual(ang, np.arctan2(200, 200), places=5, msg="Rotation angle is incorrect.")
        
if __name__ == '__main__':
    unittest.main()