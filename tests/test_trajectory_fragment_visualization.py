import unittest

from PIL import Image

from scripts.visualize_trajectory_fragments import verify_background_alignment


class TrajectoryFragmentBackgroundAlignmentTest(unittest.TestCase):
    def test_exact_original_size_is_aligned_without_transform(self):
        image = Image.new("RGB", (10, 20), (1, 2, 3))
        report = verify_background_alignment(
            image, {"image_size": [10, 20]})
        self.assertEqual(report["status"], "aligned_original_size")
        self.assertEqual(report["coordinate_offset_xy"], [0, 0])
        self.assertEqual(report["coordinate_scale_xy"], [1.0, 1.0])

    def test_uniform_right_bottom_canvas_is_top_left_padding(self):
        image = Image.new("RGB", (32, 32), (0, 0, 0))
        image.paste(
            Image.new("RGB", (10, 20), (10, 20, 30)),
            (0, 0),
        )
        report = verify_background_alignment(
            image,
            {"image_size": [10, 20], "canvas_size": 32},
        )
        self.assertEqual(report["status"], "aligned_top_left_padding")
        self.assertEqual(report["padding_pixel"], [0, 0, 0])
        self.assertEqual(report["coordinate_offset_xy"], [0, 0])

    def test_nonuniform_larger_image_is_rejected_instead_of_guessed(self):
        image = Image.new("RGB", (32, 32), (0, 0, 0))
        image.paste(
            Image.new("RGB", (10, 20), (10, 20, 30)),
            (0, 0),
        )
        image.putpixel((31, 31), (255, 255, 255))
        with self.assertRaisesRegex(ValueError, "refusing to guess"):
            verify_background_alignment(
                image,
                {"image_size": [10, 20], "canvas_size": 32},
            )


if __name__ == "__main__":
    unittest.main()
