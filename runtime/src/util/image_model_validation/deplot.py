# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import click
from util import display_image
from util import initialize_triton_client
from util import load_and_preprocess_image
from util import perform_inference
from util import prepare_input_tensor
from util import prepare_output_tensor
from util import print_output_results
from util import validate_output

logger = logging.getLogger(__name__)


@click.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.option("--display", is_flag=True, help="Display the image before sending it for inference.")
def main(image_path, display):
    # Configuration
    url = "localhost:8004"
    model_name = "deplot"
    batch_size = 1
    target_img_size = (1024, 1024)

    # Workflow
    triton_client = initialize_triton_client(url)
    resized_image, input_data = load_and_preprocess_image(image_path, target_img_size)

    if display:
        display_image(resized_image)

    input_dims = input_data.shape[1:]  # Exclude batch dimension
    logger.info(f"Detected input dimensions: {input_dims}")

    inputs = prepare_input_tensor(input_data)
    outputs = prepare_output_tensor()

    results = perform_inference(triton_client, model_name, inputs, outputs)
    output_data = results.as_numpy("output")

    validate_output(output_data, batch_size)
    print_output_results(output_data)


if __name__ == "__main__":
    main()
