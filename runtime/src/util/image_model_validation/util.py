# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys

import cv2
import numpy as np
import tritonclient.grpc as grpcclient


def resize_image(image, target_img_size):
    """
    Resize and pad an image to the target image size.
    """
    w, h, _ = np.array(image).shape

    if target_img_size is not None:  # Resize + Pad
        r = min(target_img_size[0] / w, target_img_size[1] / h)
        image = cv2.resize(
            image,
            (int(h * r), int(w * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        image = np.pad(
            image,
            ((0, target_img_size[0] - image.shape[0]), (0, target_img_size[1] - image.shape[1]), (0, 0)),
            mode="constant",
            constant_values=114,
        )
    return image


def load_and_preprocess_image(image_path, target_img_size):
    """
    Load and preprocess an image from the specified path, resizing and padding it to the target size.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from path: {image_path}")

    # Resize and pad the image to the target size
    resized_image = resize_image(image, target_img_size)

    # Normalize the image (assuming model expects normalized input)
    normalized_image = resized_image.astype(np.float32) / 255.0

    # Expand dimensions to match the model's input shape
    input_data = np.expand_dims(normalized_image, axis=0)  # Add batch dimension
    return resized_image, input_data


def validate_output(output_data, expected_batch_size):
    """
    Validate the size of the output data.
    """
    if len(output_data) != expected_batch_size:
        raise ValueError(f"Output size {len(output_data)} does not match expected batch size {expected_batch_size}.")
    print(f"Output size is valid: {len(output_data)}")


def initialize_triton_client(url):
    try:
        return grpcclient.InferenceServerClient(url=url)
    except Exception as e:
        print(f"Failed to establish connection to Triton server: {str(e)}")
        sys.exit(1)


def display_image(image):
    cv2.imshow("Input Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def print_output_results(output_data):
    print("Output Results:")
    for i, result in enumerate(output_data):
        print(f"Result {i}: {result}")


def prepare_input_tensor(input_data):
    inputs = []
    inputs.append(grpcclient.InferInput("input", input_data.shape, "FP32"))
    inputs[0].set_data_from_numpy(input_data.astype(np.float32))
    return inputs


def prepare_output_tensor():
    outputs = []
    outputs.append(grpcclient.InferRequestedOutput("output"))
    return outputs


def perform_inference(triton_client, model_name, inputs, outputs):
    try:
        return triton_client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
    except Exception as e:
        print(f"Inference failed: {str(e)}")
        sys.exit(1)
