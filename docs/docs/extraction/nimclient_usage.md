# NimClient Usage Guide

The `NimClient` class provides a unified interface for connecting to and interacting with NVIDIA Inference Microservices (NIMs). This guide demonstrates how to create custom NIM integrations for use in NV-Ingest pipelines and User Defined Functions (UDFs).

## Overview

The NimClient architecture consists of two main components:

1. **NimClient**: The client class that handles communication with NIM endpoints via gRPC or HTTP protocols
2. **ModelInterface**: An abstract base class that defines how to format input data, parse output responses, and process inference results for specific models

## Quick Start

### Basic NimClient Creation

```python
from nv_ingest_api.util.nim import create_inference_client
from nv_ingest_api.internal.primitives.nim import ModelInterface

# Create a custom model interface (see examples below)
model_interface = MyCustomModelInterface()

# Define endpoints (gRPC, HTTP)
endpoints = ("grpc://my-nim-service:8001", "http://my-nim-service:8000")

# Create the client
client = create_inference_client(
    endpoints=endpoints,
    model_interface=model_interface,
    auth_token="your-ngc-api-key",  # Optional
    infer_protocol="grpc",          # Optional: "grpc" or "http"
    timeout=120.0,                  # Optional: request timeout
    max_retries=5                   # Optional: retry attempts
)

# Perform inference
data = {"input": "your input data"}
results = client.infer(data, model_name="your-model-name")
```

### Using Environment Variables

```python
import os
from nv_ingest_api.util.nim import create_inference_client

# Use environment variables for configuration
auth_token = os.getenv("NGC_API_KEY")
grpc_endpoint = os.getenv("NIM_GRPC_ENDPOINT", "grpc://localhost:8001")
http_endpoint = os.getenv("NIM_HTTP_ENDPOINT", "http://localhost:8000")

client = create_inference_client(
    endpoints=(grpc_endpoint, http_endpoint),
    model_interface=model_interface,
    auth_token=auth_token
)
```

## Creating Custom Model Interfaces

To integrate a new NIM, you need to create a custom `ModelInterface` subclass that implements the required methods.

### Basic Model Interface Template

```python
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from nv_ingest_api.internal.primitives.nim import ModelInterface

class MyCustomModelInterface(ModelInterface):
    """
    Custom model interface for My Custom NIM.
    """
    
    def __init__(self, model_name: str = "my-custom-model"):
        """Initialize the model interface."""
        self.model_name = model_name
    
    def name(self) -> str:
        """Return the name of this model interface."""
        return "MyCustomModel"
    
    def prepare_data_for_inference(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare and validate input data before formatting.
        
        Parameters
        ----------
        data : dict
            Raw input data
            
        Returns
        -------
        dict
            Validated and prepared data
        """
        # Validate required fields
        if "input_text" not in data:
            raise KeyError("Input data must include 'input_text'")
        
        # Ensure input is in the expected format
        if not isinstance(data["input_text"], str):
            raise ValueError("input_text must be a string")
        
        return data
    
    def format_input(
        self, 
        data: Dict[str, Any], 
        protocol: str, 
        max_batch_size: int, 
        **kwargs
    ) -> Tuple[List[Any], List[Dict[str, Any]]]:
        """
        Format input data for the specified protocol.
        
        Parameters
        ----------
        data : dict
            Prepared input data
        protocol : str
            Communication protocol ("grpc" or "http")
        max_batch_size : int
            Maximum batch size for processing
        **kwargs : dict
            Additional parameters
            
        Returns
        -------
        tuple
            (formatted_batches, batch_data_list)
        """
        if protocol == "http":
            return self._format_http_input(data, max_batch_size, **kwargs)
        elif protocol == "grpc":
            return self._format_grpc_input(data, max_batch_size, **kwargs)
        else:
            raise ValueError("Invalid protocol. Must be 'grpc' or 'http'")
    
    def _format_http_input(
        self, 
        data: Dict[str, Any], 
        max_batch_size: int, 
        **kwargs
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Format input for HTTP protocol."""
        input_text = data["input_text"]
        
        # Create HTTP payload
        payload = {
            "model": kwargs.get("model_name", self.model_name),
            "input": input_text,
            "max_tokens": kwargs.get("max_tokens", 512),
            "temperature": kwargs.get("temperature", 0.7),
        }
        
        # Return as single batch
        return [payload], [{"original_input": input_text}]
    
    def _format_grpc_input(
        self, 
        data: Dict[str, Any], 
        max_batch_size: int, 
        **kwargs
    ) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
        """Format input for gRPC protocol."""
        input_text = data["input_text"]
        
        # Convert to numpy array for gRPC
        text_array = np.array([[input_text.encode("utf-8")]], dtype=np.object_)
        
        return [text_array], [{"original_input": input_text}]
    
    def parse_output(
        self, 
        response: Any, 
        protocol: str, 
        data: Optional[Dict[str, Any]] = None, 
        **kwargs
    ) -> Any:
        """
        Parse the raw model response.
        
        Parameters
        ----------
        response : Any
            Raw response from the model
        protocol : str
            Communication protocol used
        data : dict, optional
            Original batch data
        **kwargs : dict
            Additional parameters
            
        Returns
        -------
        Any
            Parsed response data
        """
        if protocol == "http":
            return self._parse_http_response(response)
        elif protocol == "grpc":
            return self._parse_grpc_response(response)
        else:
            raise ValueError("Invalid protocol. Must be 'grpc' or 'http'")
    
    def _parse_http_response(self, response: Dict[str, Any]) -> str:
        """Parse HTTP response."""
        if isinstance(response, dict):
            # Extract the generated text from response
            if "choices" in response:
                return response["choices"][0].get("text", "")
            elif "output" in response:
                return response["output"]
            else:
                raise RuntimeError("Unexpected response format")
        return str(response)
    
    def _parse_grpc_response(self, response: np.ndarray) -> str:
        """Parse gRPC response."""
        if isinstance(response, np.ndarray):
            # Decode bytes response
            return response.flatten()[0].decode("utf-8")
        return str(response)
    
    def process_inference_results(
        self, 
        output: Any, 
        protocol: str, 
        **kwargs
    ) -> Any:
        """
        Post-process the parsed inference results.
        
        Parameters
        ----------
        output : Any
            Parsed output from parse_output
        protocol : str
            Communication protocol used
        **kwargs : dict
            Additional parameters
            
        Returns
        -------
        Any
            Final processed results
        """
        # Apply any final processing (e.g., filtering, formatting)
        if isinstance(output, str):
            return output.strip()
        return output
```

## Real-World Examples

### Text Generation Model Interface

```python
class TextGenerationModelInterface(ModelInterface):
    """Interface for text generation NIMs (e.g., LLaMA, GPT-style models)."""
    
    def name(self) -> str:
        return "TextGeneration"
    
    def prepare_data_for_inference(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if "prompt" not in data:
            raise KeyError("Input data must include 'prompt'")
        return data
    
    def format_input(self, data: Dict[str, Any], protocol: str, max_batch_size: int, **kwargs):
        prompt = data["prompt"]
        
        if protocol == "http":
            payload = {
                "model": kwargs.get("model_name", "llama-2-7b-chat"),
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": kwargs.get("max_tokens", 512),
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9),
                "stream": False
            }
            return [payload], [{"prompt": prompt}]
        else:
            raise ValueError("Only HTTP protocol supported for this model")
    
    def parse_output(self, response: Any, protocol: str, data: Optional[Dict[str, Any]] = None, **kwargs):
        if protocol == "http" and isinstance(response, dict):
            choices = response.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
        return str(response)
    
    def process_inference_results(self, output: Any, protocol: str, **kwargs):
        return output.strip() if isinstance(output, str) else output
```

### Image Analysis Model Interface

```python
import base64
from nv_ingest_api.util.image_processing.transforms import numpy_to_base64

class ImageAnalysisModelInterface(ModelInterface):
    """Interface for image analysis NIMs (e.g., vision models)."""
    
    def name(self) -> str:
        return "ImageAnalysis"
    
    def prepare_data_for_inference(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if "images" not in data:
            raise KeyError("Input data must include 'images'")
        
        # Ensure images is a list
        if not isinstance(data["images"], list):
            data["images"] = [data["images"]]
        
        return data
    
    def format_input(self, data: Dict[str, Any], protocol: str, max_batch_size: int, **kwargs):
        images = data["images"]
        prompt = data.get("prompt", "Describe this image.")
        
        # Convert images to base64 if needed
        base64_images = []
        for img in images:
            if isinstance(img, np.ndarray):
                base64_images.append(numpy_to_base64(img))
            elif isinstance(img, str) and img.startswith("data:image"):
                # Already base64 encoded
                base64_images.append(img.split(",")[1])
            else:
                base64_images.append(str(img))
        
        # Batch images
        batches = [base64_images[i:i + max_batch_size] 
                  for i in range(0, len(base64_images), max_batch_size)]
        
        payloads = []
        batch_data_list = []
        
        for batch in batches:
            if protocol == "http":
                messages = []
                for img_b64 in batch:
                    messages.append({
                        "role": "user",
                        "content": f'{prompt} <img src="data:image/png;base64,{img_b64}" />'
                    })
                
                payload = {
                    "model": kwargs.get("model_name", "llava-1.5-7b-hf"),
                    "messages": messages,
                    "max_tokens": kwargs.get("max_tokens", 512),
                    "temperature": kwargs.get("temperature", 0.1)
                }
                payloads.append(payload)
                batch_data_list.append({"images": batch, "prompt": prompt})
        
        return payloads, batch_data_list
    
    def parse_output(self, response: Any, protocol: str, data: Optional[Dict[str, Any]] = None, **kwargs):
        if protocol == "http" and isinstance(response, dict):
            choices = response.get("choices", [])
            return [choice.get("message", {}).get("content", "") for choice in choices]
        return [str(response)]
    
    def process_inference_results(self, output: Any, protocol: str, **kwargs):
        if isinstance(output, list):
            return [result.strip() for result in output]
        return output
```

## Using NimClient in UDFs

### Basic UDF with NimClient

```python
from nv_ingest_api.internal.primitives.control_message import IngestControlMessage
from nv_ingest_api.util.nim import create_inference_client
import os

def analyze_document_with_nim(control_message: IngestControlMessage) -> IngestControlMessage:
    """UDF that uses a custom NIM to analyze document content."""
    
    # Create NIM client
    model_interface = TextGenerationModelInterface()
    client = create_inference_client(
        endpoints=(
            os.getenv("ANALYSIS_NIM_GRPC", "grpc://analysis-nim:8001"),
            os.getenv("ANALYSIS_NIM_HTTP", "http://analysis-nim:8000")
        ),
        model_interface=model_interface,
        auth_token=os.getenv("NGC_API_KEY"),
        infer_protocol="http"
    )
    
    # Get the document DataFrame
    df = control_message.get_payload()
    
    # Process each document
    for idx, row in df.iterrows():
        if row.get("content"):
            # Prepare analysis prompt
            prompt = f"Analyze the following document content and provide a summary: {row['content'][:1000]}"
            
            # Perform inference
            try:
                results = client.infer(
                    data={"prompt": prompt},
                    model_name="llama-2-7b-chat",
                    max_tokens=256,
                    temperature=0.3
                )
                
                # Add analysis to metadata
                if results:
                    analysis = results[0] if isinstance(results, list) else results
                    df.at[idx, "custom_analysis"] = analysis
                    
            except Exception as e:
                print(f"NIM inference failed: {e}")
                df.at[idx, "custom_analysis"] = "Analysis failed"
    
    # Update the control message with processed data
    control_message.payload(df)
    return control_message
```

### Advanced UDF with Batching

```python
def batch_image_analysis_udf(control_message: IngestControlMessage) -> IngestControlMessage:
    """UDF that performs batched image analysis using NIM."""
    
    # Create image analysis client
    model_interface = ImageAnalysisModelInterface()
    client = create_inference_client(
        endpoints=(
            os.getenv("VISION_NIM_GRPC", "grpc://vision-nim:8001"),
            os.getenv("VISION_NIM_HTTP", "http://vision-nim:8000")
        ),
        model_interface=model_interface,
        auth_token=os.getenv("NGC_API_KEY")
    )
    
    df = control_message.get_payload()
    
    # Collect all images for batch processing
    image_rows = []
    images = []
    
    for idx, row in df.iterrows():
        if "image_data" in row and row["image_data"]:
            image_rows.append(idx)
            images.append(row["image_data"])
    
    if images:
        try:
            # Batch process all images
            results = client.infer(
                data={
                    "images": images,
                    "prompt": "Describe the content and key elements in this image."
                },
                model_name="llava-1.5-7b-hf",
                max_tokens=200
            )
            
            # Apply results back to DataFrame
            for idx, result in zip(image_rows, results):
                df.at[idx, "image_description"] = result
                
        except Exception as e:
            print(f"Batch image analysis failed: {e}")
            for idx in image_rows:
                df.at[idx, "image_description"] = "Analysis failed"
    
    control_message.payload(df)
    return control_message
```

## Configuration and Best Practices

### Environment Variables

Set these environment variables for your NIM endpoints:

```bash
# NIM endpoints
export MY_NIM_GRPC_ENDPOINT="grpc://my-nim-service:8001"
export MY_NIM_HTTP_ENDPOINT="http://my-nim-service:8000"

# Authentication
export NGC_API_KEY="your-ngc-api-key"

# Optional: timeouts and retries
export NIM_TIMEOUT=120
export NIM_MAX_RETRIES=5
```

### Performance Optimization

1. **Use gRPC when possible**: Generally faster than HTTP for high-throughput scenarios
2. **Batch processing**: Process multiple items together to reduce overhead
3. **Connection reuse**: Create NimClient instances once and reuse them
4. **Appropriate timeouts**: Set reasonable timeouts based on your model's response time
5. **Error handling**: Always handle inference failures gracefully

### Error Handling

```python
def robust_nim_udf(control_message: IngestControlMessage) -> IngestControlMessage:
    """UDF with comprehensive error handling."""
    
    try:
        client = create_inference_client(
            endpoints=(grpc_endpoint, http_endpoint),
            model_interface=model_interface,
            auth_token=auth_token,
            timeout=60.0,
            max_retries=3
        )
    except Exception as e:
        print(f"Failed to create NIM client: {e}")
        return control_message
    
    df = control_message.get_payload()
    
    for idx, row in df.iterrows():
        try:
            results = client.infer(data=input_data, model_name="my-model")
            df.at[idx, "nim_result"] = results
        except TimeoutError:
            print(f"NIM request timed out for row {idx}")
            df.at[idx, "nim_result"] = "timeout"
        except Exception as e:
            print(f"NIM inference failed for row {idx}: {e}")
            df.at[idx, "nim_result"] = "error"
    
    control_message.payload(df)
    return control_message
```

## Troubleshooting

### Common Issues

1. **Connection Errors**: Verify NIM service is running and endpoints are correct
2. **Authentication Failures**: Check NGC_API_KEY is valid and properly set
3. **Timeout Errors**: Increase timeout values or check NIM service performance
4. **Format Errors**: Ensure your ModelInterface formats data correctly for your NIM
5. **Memory Issues**: Use appropriate batch sizes to avoid memory exhaustion

### Debugging Tips

```python
import logging

# Enable debug logging
logging.getLogger("nv_ingest_api.internal.primitives.nim").setLevel(logging.DEBUG)

# Test your model interface separately
model_interface = MyCustomModelInterface()
test_data = {"input": "test"}

# Test data preparation
prepared = model_interface.prepare_data_for_inference(test_data)
print(f"Prepared data: {prepared}")

# Test input formatting
formatted, batch_data = model_interface.format_input(prepared, "http", 1)
print(f"Formatted input: {formatted}")
```

## Additional Resources

- [NV-Ingest UDF Documentation](user_defined_functions.md)
- [NVIDIA NIM Documentation](https://docs.nvidia.com/nim/)
- [Pipeline Configuration Guide](../configuration/pipeline_config.md)

For more examples and advanced usage patterns, see the existing model interfaces in:
`api/src/nv_ingest_api/internal/primitives/nim/model_interface/`
