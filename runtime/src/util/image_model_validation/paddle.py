# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import click
import numpy as np
from util import display_image
from util import initialize_triton_client
from util import load_and_preprocess_image
from util import perform_inference
from util import prepare_input_tensor
from util import prepare_output_tensor
from util import print_output_results
from util import validate_output


@click.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.option("--display", is_flag=True, help="Display the image before sending it for inference.")
def main(image_path, display):
    # Triton server URL and Model details
    url = "localhost:8010"
    model_name = "paddle"
    batch_size = 1
    target_img_size = (1024, 1024)

    # Load and preprocess image
    resized_image, input_data = load_and_preprocess_image(image_path, target_img_size)
    resized_images = np.expand_dims(resized_image, axis=0)  # Add batch dimension

    # Optionally display the image
    if display:
        display_image(resized_image)

    # Detect input dimensions from the loaded image
    input_dims = input_data.shape[1:]  # Exclude the batch dimension
    print(f"Detected input dimensions: {input_dims}")

    # Initialize Triton gRPC client
    triton_client = initialize_triton_client(url)

    # Prepare input and output tensors
    inputs = prepare_input_tensor(resized_images)
    outputs = prepare_output_tensor()

    # Call the Triton server for inference
    results = perform_inference(triton_client, model_name, inputs, outputs)

    # Get output data
    output_data = results.as_numpy("output")

    # Validate output size
    validate_output(output_data, batch_size)

    # Print the output results
    print_output_results(output_data)


if __name__ == "__main__":
    main()
