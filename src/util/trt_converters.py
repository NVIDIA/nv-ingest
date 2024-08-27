# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from typing import List

import click
import tensorrt as trt
import torch
from pydantic import BaseModel
from pydantic import ValidationError


class IOConfig(BaseModel):
    name: str
    dtype: str
    shape: List[int]


class ModelIOConfig(BaseModel):
    inputs: List[IOConfig]
    outputs: List[IOConfig]


def generate_preferred_batch_sizes(max_batch_size: int) -> str:
    sizes = []
    size = 1
    while size <= max_batch_size:
        sizes.append(size)
        size *= 2
    return ",".join(map(str, sizes))


@click.command()
@click.option(
    "--ckpt-path", required=True, type=click.Path(), help="Path to the PyTorch model checkpoint file (state_dict)."
)
@click.option("--device", default="cpu", type=click.Choice(["cpu", "cuda"]), help="Device to load the model on.")
@click.option("--dynamic-batching", is_flag=True, help="Enable dynamic batching in Triton.")
@click.option("--engine-path", required=True, type=click.Path(), help="Path to save the TensorRT engine file.")
@click.option("--generate-onnx", is_flag=True, help="Generate ONNX model.")
@click.option("--generate-triton-config", is_flag=True, help="Generate Triton configuration.")
@click.option("--generate-trt", is_flag=True, help="Generate TensorRT engine.")
@click.option(
    "--input-size", type=(int, int), default=(None, 128), help="Input size for the model (batch size, sequence length)."
)
@click.option("--max-batch-size", default=128, help="Maximum batch size for Triton.")
@click.option("--max-queue-delay", default=100, help="Max queue delay in microseconds for Triton dynamic batching.")
@click.option("--memory-pool-limit", default=1 << 30, help="Memory pool limit for TensorRT builder (in bytes).")
@click.option("--model-class-name", required=True, help="Name of the model class in the script.")
@click.option("--model-io-config", required=True, type=str, help="JSON string defining model inputs and outputs.")
@click.option("--model-name", required=True, help="Name of the model.")
@click.option(
    "--model-script-path", required=True, type=click.Path(), help="Path to the PyTorch model script (Python file)."
)
@click.option("--onnx-path", required=True, type=click.Path(), help="Path to save the ONNX model file.")
@click.option("--overwrite", is_flag=True, help="Overwrite existing ONNX and TensorRT models.")
@click.option("--preferred-batch-sizes", type=str, default="", help="Comma-separated list of preferred batch sizes.")
@click.option("--triton-inputs", type=str, help="Comma-separated list of input names to include in the Triton config.")
@click.option("--triton-repo", required=True, type=click.Path(), help="Path to Triton model repository.")
@click.option("--use-fp16", is_flag=True, help="Enable FP16 precision.")
@click.option("--use-int8", is_flag=True, help="Enable INT8 precision.")
def convert_model(
    ckpt_path,
    device,
    dynamic_batching,
    engine_path,
    generate_onnx,
    generate_triton_config,
    generate_trt,
    input_size,
    max_batch_size,
    max_queue_delay,
    memory_pool_limit,
    model_class_name,
    model_io_config,
    model_name,
    model_script_path,
    onnx_path,
    overwrite,
    preferred_batch_sizes,
    triton_inputs,
    triton_repo,
    use_fp16,
    use_int8,
):
    """
    Convert a PyTorch model to ONNX and then to TensorRT engine, and set up Triton Inference Server model repository.
    """
    try:
        io_config = ModelIOConfig.parse_raw(model_io_config)
    except ValidationError as e:
        raise click.ClickException(f"Invalid model IO configuration: {e}")

    # Load the PyTorch model
    model = load_pytorch_model(model_script_path, model_class_name, ckpt_path, device)

    # Check if model directories exist in Triton repository
    model_dir_onnx = os.path.join(triton_repo, f"{model_name}_onnx")
    model_dir_trt = os.path.join(triton_repo, f"{model_name}_trt")

    input_names_list = [io.name for io in io_config.inputs]
    input_types_list = [io.dtype for io in io_config.inputs]
    output_names_list = [io.name for io in io_config.outputs]
    output_types_list = [io.dtype for io in io_config.outputs]

    # Parse Triton inputs
    triton_inputs_list = triton_inputs.split(",") if triton_inputs else input_names_list

    # Generate default preferred batch sizes if not provided
    if not preferred_batch_sizes:
        preferred_batch_sizes = generate_preferred_batch_sizes(max_batch_size)

    if generate_onnx:
        if not os.path.exists(onnx_path) or overwrite:
            # Convert PyTorch model to ONNX
            convert_to_onnx(
                model,
                onnx_path,
                input_size,
                input_names_list,
                input_types_list,
                output_names_list,
                output_types_list,
                device,
            )
        else:
            click.echo(f"ONNX model already exists at {onnx_path}, skipping ONNX conversion.")

    if generate_trt:
        if not os.path.exists(engine_path) or overwrite:
            # Convert ONNX model to TensorRT engine
            convert_to_trt(onnx_path, engine_path, use_fp16, use_int8, memory_pool_limit, max_batch_size)
        else:
            click.echo(f"TensorRT engine already exists at {engine_path}, skipping TensorRT conversion.")

    if generate_triton_config:
        if not os.path.exists(model_dir_onnx):
            os.makedirs(os.path.join(model_dir_onnx, "1"), exist_ok=True)
        if not os.path.exists(model_dir_trt):
            os.makedirs(os.path.join(model_dir_trt, "1"), exist_ok=True)

        config_path_onnx = os.path.join(model_dir_onnx, "config.pbtxt")
        config_path_trt = os.path.join(model_dir_trt, "config.pbtxt")

        if generate_onnx and (not os.path.exists(config_path_onnx) or overwrite):
            # Create Triton configuration file for ONNX
            create_triton_config(
                model_name + "_onnx",
                model_dir_onnx,
                input_size,
                dynamic_batching,
                max_batch_size,
                preferred_batch_sizes,
                max_queue_delay,
                is_trt=False,
                io_config=io_config,
                triton_inputs_list=triton_inputs_list,
            )
        else:
            click.echo(f"Triton config file already exists at {config_path_onnx}, skipping config creation for ONNX.")

        if generate_trt and (not os.path.exists(config_path_trt) or overwrite):
            # Create Triton configuration file for TensorRT
            create_triton_config(
                model_name + "_trt",
                model_dir_trt,
                input_size,
                dynamic_batching,
                max_batch_size,
                preferred_batch_sizes,
                max_queue_delay,
                is_trt=True,
                io_config=io_config,
                triton_inputs_list=triton_inputs_list,
            )
        else:
            click.echo(
                f"Triton config file already exists at {config_path_trt}, skipping config creation for TensorRT."
            )

    # Move files to Triton model repository if they are not already there
    if generate_onnx and (not os.path.exists(os.path.join(model_dir_onnx, "1", "model.onnx")) or overwrite):
        if os.path.exists(os.path.join(model_dir_onnx, "1", "model.onnx")) and overwrite:
            os.remove(os.path.join(model_dir_onnx, "1", "model.onnx"))
        os.rename(onnx_path, os.path.join(model_dir_onnx, "1", "model.onnx"))
    else:
        click.echo("ONNX model already present in Triton repository, skipping move.")

    if generate_trt and (not os.path.exists(os.path.join(model_dir_trt, "1", "model.plan")) or overwrite):
        if os.path.exists(os.path.join(model_dir_trt, "1", "model.plan")) and overwrite:
            os.remove(os.path.join(model_dir_trt, "1", "model.plan"))
        os.rename(engine_path, os.path.join(model_dir_trt, "1", "model.plan"))
    else:
        click.echo("TensorRT engine already present in Triton repository, skipping move.")


def load_pytorch_model(model_script_path, model_class_name, ckpt_path, device):
    # Load the PyTorch model from the script file and state_dict
    click.echo(f"Loading PyTorch model from {model_script_path} and {ckpt_path}")
    model_script_dir = os.path.dirname(model_script_path)
    model_script_name = os.path.basename(model_script_path).replace(".py", "")

    # Add the model script directory to the system path
    import sys

    sys.path.append(model_script_dir)

    # Import the model script module
    model_script = __import__(model_script_name)

    # Dynamically get the model class from the script
    model_class = getattr(model_script, model_class_name)

    # Instantiate the model
    model = model_class()
    model = model.to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    if device == "cuda":
        model = torch.nn.DataParallel(model)
    model.eval()

    return model


def convert_to_onnx(model, onnx_path, input_size, input_names, input_types, output_names, output_types, device):
    # Convert PyTorch model to ONNX format
    click.echo(f"Converting model to ONNX and saving to {onnx_path}")

    # Prepare dummy input based on specified data types
    batch_size, seq_length = input_size
    dummy_inputs = {}
    for name, dtype in zip(input_names, input_types):
        if dtype == "float32":
            dummy_inputs[name] = torch.randn(batch_size, seq_length).float().to(device)
        elif dtype == "int64":
            dummy_inputs[name] = torch.randint(0, 1000, (batch_size, seq_length)).long().to(device)
        # Add other data types as needed

    # Handle DataParallel case
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    dynamic_axes = {name: {0: "batch_size"} for name in input_names}
    dynamic_axes.update({name: {0: "batch_size"} for name in output_names})

    print(input_names)
    print(output_names)
    print(dummy_inputs)
    print(dynamic_axes)

    # Note to self, opsets 11-17 appear to work.
    torch.onnx.export(
        model,
        tuple(dummy_inputs.values()),
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )


def convert_to_trt(onnx_path, engine_path, use_fp16, use_int8, memory_pool_limit, max_batch_size):
    # Convert ONNX model to TensorRT engine
    click.echo(f"Converting ONNX model to TensorRT engine and saving to {engine_path}")

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX model
    with open(onnx_path, "rb") as model_file:
        if not parser.parse(model_file.read()):
            for error in range(parser.num_errors):
                click.echo(f"TensorRT ONNX parser error: {parser.get_error(error)}")
            return

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, memory_pool_limit)  # Set memory pool limit
    config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
    config.set_flag(trt.BuilderFlag.TF32)
    if use_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if use_int8:
        config.set_flag(trt.BuilderFlag.INT8)

    profile = builder.create_optimization_profile()
    # Ensure the minimum, optimum, and maximum dimensions follow the specified conditions
    profile.set_shape("input_ids", (1, 128), (8, 128), (max_batch_size, 128))
    profile.set_shape("input_mask", (1, 128), (8, 128), (max_batch_size, 128))
    profile.set_shape("input_token_type_ids", (1, 128), (8, 128), (max_batch_size, 128))
    config.add_optimization_profile(profile)

    # Build the engine
    engine = builder.build_serialized_network(network, config)
    with open(engine_path, "wb") as f:
        f.write(engine)


def create_triton_config(
    model_name,
    model_dir,
    input_size,
    dynamic_batching,
    max_batch_size,
    preferred_batch_sizes,
    max_queue_delay,
    is_trt,
    io_config,
    triton_inputs_list,
):
    # Create Triton configuration file (config.pbtxt)
    config_path = os.path.join(model_dir, "config.pbtxt")
    click.echo(f"Creating Triton configuration file at {config_path}")

    preferred_batch_sizes_list = [int(x) for x in preferred_batch_sizes.split(",")]

    # Mapping from common data types to Triton data types
    dtype_map = {
        "float32": "TYPE_FP32",
        "int64": "TYPE_INT64",
        "int32": "TYPE_INT32",
        # Add other necessary mappings as needed
    }

    inputs_config = [
        {"name": inp.name, "data_type": dtype_map[inp.dtype], "dims": inp.shape}
        for inp in io_config.inputs
        if inp.name in triton_inputs_list
    ]

    outputs_config = [
        {"name": out.name, "data_type": dtype_map[out.dtype], "dims": out.shape} for out in io_config.outputs
    ]

    # Generate input and output configurations
    inputs_str = ",\n".join(
        [
            f"""
  {{
    name: "{inp['name']}"
    data_type: {inp['data_type']}
    dims: {inp['dims']}
  }}"""
            for inp in inputs_config
        ]
    )

    outputs_str = ",\n".join(
        [
            f"""
  {{
    name: "{out['name']}"
    data_type: {out['data_type']}
    dims: {out['dims']}
  }}"""
            for out in outputs_config
        ]
    )

    # Generate the final configuration
    config = f"""
name: "{model_name}"
platform: "{"onnxruntime_onnx" if not is_trt else "tensorrt_plan"}"
max_batch_size: {max_batch_size}
input [
{inputs_str}
]
output [
{outputs_str}
]
"""

    if dynamic_batching:
        config += f"""
dynamic_batching {{
    preferred_batch_size: {preferred_batch_sizes_list}
    max_queue_delay_microseconds: {max_queue_delay}
}}
"""

    with open(config_path, "w") as f:
        f.write(config)


if __name__ == "__main__":
    convert_model()
