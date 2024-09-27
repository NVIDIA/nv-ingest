import numpy as np

from nv_ingest.util.image_processing.transforms import numpy_to_base64


def test_numpy_to_base64_valid_rgba_image():
    array = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
    result = numpy_to_base64(array)

    assert isinstance(result, str)
    assert len(result) > 0


def test_numpy_to_base64_valid_rgb_image():
    array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = numpy_to_base64(array)

    assert isinstance(result, str)
    assert len(result) > 0


def test_numpy_to_base64_grayscale_redundant_axis():
    array = np.random.randint(0, 255, (100, 100, 1), dtype=np.uint8)
    result = numpy_to_base64(array)

    assert isinstance(result, str)
    assert len(result) > 0
