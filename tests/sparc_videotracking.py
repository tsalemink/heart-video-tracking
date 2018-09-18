import os
import unittest

from sparc.videotracking.processing import Processing

here = os.path.dirname(__file__)
resource_dir = os.path.join(here, 'resources')


class ProcessingTestCase(unittest.TestCase):

    def test_process_image(self):
        image_file = os.path.join(resource_dir, 'pig_heart_1.jpg')
        processor = Processing()
        processor.read_image(image_file)
        processor.filter_and_threshold()
        processor.mask_and_image((603, 146, 129, 174))
        image_points, dst = processor.feature_detect()

        self.assertEqual(3, len(image_points))
        self.assertEqual(8, len(dst))

        pt = image_points[0].pt
        self.assertAlmostEqual(647, pt[0], places=0)
        self.assertAlmostEqual(190, pt[1], places=0)


if __name__ == '__main__':
    unittest.main()