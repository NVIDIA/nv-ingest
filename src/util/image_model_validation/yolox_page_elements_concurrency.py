# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import time
import csv
import click
import signal
import os
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from typing import Callable

# Make shared benchmarking helpers importable when running this script directly
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UTIL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if UTIL_DIR not in sys.path:
    sys.path.insert(0, UTIL_DIR)

from benchmarking.common_components import (  # noqa: E402
    essential_headers,
    encode_image_file_to_data_url,
    load_image_numpy,
    build_http_payload_from_data_url,
    http_post_with_timing,
    yolox_grpc_infer_with_timing,
)

# Global variable to track if the test is running
running = True


def signal_handler(sig, frame):
    """Handle Ctrl+C to gracefully exit the running tests"""
    global running
    print("\nGracefully stopping... (This may take a few seconds)")
    running = False


# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)


@click.command()
@click.option(
    "--image-path", required=False, type=click.Path(exists=True), help="Path to the image file to be encoded and sent."
)
@click.option(
    "--url",
    default="http://localhost:8000/v1/infer",
    help="Endpoint URL for HTTP (e.g., http://host:8000/v1/infer) or gRPC address for Triton (e.g., 127.0.0.1:8001).",
)
@click.option(
    "--protocol",
    type=click.Choice(["http", "grpc"], case_sensitive=False),
    default="http",
    help="Protocol to use for requests.",
)
@click.option("--min-concurrency", default=1, help="Minimum number of concurrent connections to test.", type=int)
@click.option("--max-concurrency", default=64, help="Maximum number of concurrent connections to test.", type=int)
@click.option("--concurrency-step", default=2, help="Step size for concurrency values (multiplier).", type=float)
@click.option("--start-batch", default=1, help="Starting batch size.", type=int)
@click.option("--max-batch", default=128, help="Maximum batch size to test.", type=int)
@click.option("--batch-step", default=2, help="Step size for batch size values (multiplier).", type=float)
@click.option("--timeout", default=5, help="Timeout (in seconds) for each request.", type=int)
@click.option(
    "--stability-duration", default=60, help="Duration (in seconds) to test each configuration for stability.", type=int
)
@click.option("--auth-token", default=None, help="Authentication token for the Authorization header (Bearer token).")
@click.option(
    "--custom-headers", default="", help='Additional headers as a JSON string, e.g. \'{"X-Custom": "value"}\'.'
)
@click.option("--output-file", default="yolox_throughput_results.csv", help="CSV file to write results to.", type=str)
@click.option("--visualize", is_flag=True, help="Generate Plotly visualizations of the results.")
@click.option(
    "--visualize-only", type=click.Path(exists=False), help="Visualize an existing CSV file without running tests."
)
def optimize_throughput(
    image_path,
    url,
    protocol,
    min_concurrency,
    max_concurrency,
    concurrency_step,
    start_batch,
    max_batch,
    batch_step,
    timeout,
    stability_duration,
    auth_token,
    custom_headers,
    output_file,
    visualize,
    visualize_only,
):
    """
    Tests the YOLOX page elements service to find the optimal throughput configuration.

    For each concurrency level from min to max, this script finds the maximum stable batch size.
    A configuration is considered 'stable' if the service can run at that concurrency/batch size
    for at least the specified stability duration without receiving any 429 or 5xx responses.

    The script records response times and calculates throughput (images/second) for each
    configuration and writes all results to a CSV file.

    If --visualize-only is specified with a CSV file path, it will skip testing and only
    generate visualizations from the existing CSV file.
    """
    global running

    # Check if we're only visualizing an existing CSV file
    if visualize_only:
        if os.path.exists(visualize_only):
            click.echo(f"Visualizing existing CSV file: {visualize_only}")
            generate_visualizations(visualize_only)
            return
        else:
            click.echo(f"CSV file not found: {visualize_only}")
            return

    # Verify required parameters for testing
    if not image_path:
        click.echo("Error: --image-path is required when running tests.")
        return

    # Prepare per-protocol inputs and headers
    protocol = (protocol or "http").lower()
    if protocol == "http":
        # Prepare a data URL once for reuse across requests
        data_url = encode_image_file_to_data_url(image_path)
        headers = dict(essential_headers)
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        if custom_headers:
            try:
                extra_headers = json.loads(custom_headers)
                headers.update(extra_headers)
            except Exception as e:
                click.echo(f"Error parsing custom headers: {e}. Please provide a valid JSON string.")
                return
    elif protocol == "grpc":
        # Load numpy image once for reuse; headers unused
        image_np = load_image_numpy(image_path)
        headers = {}
    else:
        click.echo("Invalid protocol specified. Use http or grpc.")
        return

    # Prepare results array for CSV output
    results_data = []

    # CSV header
    csv_header = [
        "concurrency",
        "batch_size",
        "total_requests",
        "successful_requests",
        "avg_response_time",
        "min_response_time",
        "max_response_time",
        "throughput_images_per_sec",
        "throughput_requests_per_sec",
        "is_stable",
        "200s",
        "400s",
        "429s",
        "500s",
        "timeouts",
        "errors",
    ]

    # Initialize the CSV file with the header
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_header)

    # Test each concurrency level
    current_concurrency = min_concurrency
    max_throughput = 0
    optimal_config = {"concurrency": 0, "batch_size": 0, "throughput": 0}

    click.echo("\n=== Finding optimal throughput configuration ===")

    while current_concurrency <= max_concurrency and running:
        click.echo(f"\n=== Testing with concurrency: {current_concurrency} ===")

        # For each concurrency, find the maximum stable batch size
        current_batch = start_batch
        max_stable_batch = 0
        batch_found = False

        while current_batch <= max_batch and running and not batch_found:
            click.echo(f"  Testing batch size: {current_batch}")

            # Test this configuration for stability
            # Build the appropriate request function for this protocol
            request_fn = make_request_fn(
                protocol, url, headers, timeout, auth_token, data_url if protocol == "http" else image_np
            )

            stability_result = test_configuration_stability(
                request_fn, current_concurrency, current_batch, timeout, stability_duration
            )

            # Extract results
            is_stable = stability_result["is_stable"]
            response_times = stability_result["response_times"]
            status_counts = stability_result["status_counts"]
            total_requests = stability_result["total_requests"]
            successful_requests = stability_result["successful_requests"]

            # Calculate metrics
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0
            min_response_time = min(response_times) if response_times else 0
            max_response_time = max(response_times) if response_times else 0

            # Calculate throughput
            elapsed_time = stability_result["elapsed_time"]
            throughput_requests = total_requests / elapsed_time if elapsed_time > 0 else 0
            throughput_images = (total_requests * current_batch) / elapsed_time if elapsed_time > 0 else 0

            # Log results for this configuration
            result_row = [
                current_concurrency,
                current_batch,
                total_requests,
                successful_requests,
                avg_response_time,
                min_response_time,
                max_response_time,
                throughput_images,
                throughput_requests,
                is_stable,
                status_counts.get(200, 0),
                status_counts.get(400, 0) + status_counts.get(401, 0) + status_counts.get(403, 0),
                status_counts.get(429, 0),
                status_counts.get(500, 0) + status_counts.get(502, 0) + status_counts.get(503, 0),
                status_counts.get("timeout", 0),
                status_counts.get("error", 0),
            ]

            # Append to CSV
            with open(output_file, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(result_row)

            results_data.append(result_row)

            # Update the optimal configuration if this is stable and has better throughput
            if is_stable and throughput_images > max_throughput:
                max_throughput = throughput_images
                optimal_config = {
                    "concurrency": current_concurrency,
                    "batch_size": current_batch,
                    "throughput": throughput_images,
                }

            # If stable, continue to next batch size, otherwise we've found the limit
            if is_stable:
                # Calculate success rate
                success_rate = (successful_requests / total_requests) * 100 if total_requests > 0 else 0

                # Display success rate and average response time
                click.echo(f"    Success Rate: {success_rate:.2f}%, Avg Response Time: {avg_response_time:.4f}s")

                max_stable_batch = current_batch
                current_batch = int(current_batch * batch_step)
                if current_batch > max_batch:
                    batch_found = True
            else:
                # If the first batch size isn't stable, record it but continue with concurrency
                if current_batch == start_batch:
                    max_stable_batch = 0
                    batch_found = True
                else:
                    # We found our limit for this concurrency
                    click.echo(f"  Maximum stable batch size at concurrency {current_concurrency}: {max_stable_batch}")
                    batch_found = True

        # Print summary for this concurrency level
        click.echo(f"  Concurrency {current_concurrency}: Maximum stable batch size: {max_stable_batch}")
        if max_stable_batch > 0:
            click.echo(f"  Estimated throughput at this configuration: {throughput_images:.2f} images/sec")

        # Move to next concurrency
        current_concurrency = int(current_concurrency * concurrency_step)
        if current_concurrency > max_concurrency:
            break

    # Print final results
    click.echo("\n=== Testing Complete ===")
    click.echo(f"Results saved to: {output_file}")

    if optimal_config["throughput"] > 0:
        click.echo("\n=== Optimal Configuration ===")
        click.echo(f"Concurrency: {optimal_config['concurrency']}")
        click.echo(f"Batch Size: {optimal_config['batch_size']}")
        click.echo(f"Throughput: {optimal_config['throughput']:.2f} images/second")
    else:
        click.echo("\nNo stable configuration found. Try increasing timeout or adjusting other parameters.")

    # Generate visualizations if requested
    if visualize and os.path.exists(output_file) and len(results_data) > 0:
        generate_visualizations(output_file)


def test_configuration_stability(
    request_fn: Callable[[int], tuple],
    concurrency: int,
    batch_size: int,
    timeout: int,
    stability_duration: int,
):
    """
    Tests a specific concurrency/batch size configuration for stability.

    Returns a dictionary with test results including stability status, response times, and status counts.
    """
    global running

    # Initialize counters for this test
    start_time = time.time()
    end_time = start_time + stability_duration
    results = {
        "is_stable": True,
        "response_times": [],
        "status_counts": defaultdict(int),
        "total_requests": 0,
        "successful_requests": 0,
        "elapsed_time": 0,
    }

    # Continue testing until stability duration is reached or an issue is found
    while time.time() < end_time and results["is_stable"] and running:
        batch_results = []

        # Launch concurrent requests using a thread pool
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(request_fn, batch_size) for _ in range(concurrency)]
            for future in as_completed(futures):
                status, response_time = future.result()
                batch_results.append((status, response_time))

                # Count status codes
                results["status_counts"][status] += 1

                # Track response times for successful requests
                if isinstance(status, int) and 200 <= status < 300:
                    results["response_times"].append(response_time)
                    results["successful_requests"] += 1

                # Check for error conditions that indicate instability
                if (
                    (isinstance(status, int) and (status == 429 or status >= 500))
                    or status == "timeout"
                    or status == "error"
                ):
                    results["is_stable"] = False

                results["total_requests"] += 1

    # Calculate final elapsed time
    results["elapsed_time"] = time.time() - start_time

    return results


def make_request_fn(
    protocol: str,
    url: str,
    headers: dict,
    timeout: int,
    auth_token: str,
    prepped_input,
) -> Callable[[int], tuple]:
    """Return a callable that sends a single request for the given protocol.

    The callable signature is (batch_size) -> (status, response_time_seconds).
    """
    protocol = protocol.lower()

    if protocol == "http":

        def _fn(batch_size: int):
            payload = build_http_payload_from_data_url(prepped_input, batch_size)
            return http_post_with_timing(url, payload, timeout, headers)

        return _fn

    elif protocol == "grpc":

        def _fn(batch_size: int):
            return yolox_grpc_infer_with_timing(url, auth_token, prepped_input, batch_size, timeout)

        return _fn

    else:
        raise ValueError("Unsupported protocol")


def generate_visualizations(csv_file):
    """
    Generate Plotly visualizations from the CSV results file.

    Creates two graphs:
    1. Response time vs batch size for each concurrency level
    2. Response time vs concurrency for each batch size

    Both graphs include tooltips showing throughput when hovering over data points.
    """
    try:
        # Load the data from CSV
        df = pd.read_csv(csv_file)

        # Filter only stable configurations
        stable_df = df[df["is_stable"] == True].copy()  # noqa: E712

        if len(stable_df) == 0:
            click.echo("No stable configurations found for visualization.")
            return

        # Add custom hover text for the graphs
        stable_df["hover_text"] = stable_df.apply(
            lambda row: f"Concurrency: {row['concurrency']}<br>"
            + f"Batch Size: {row['batch_size']}<br>"
            + f"Response Time: {row['avg_response_time']:.4f}s<br>"
            + f"Throughput: {row['throughput_images_per_sec']:.2f} img/s",
            axis=1,
        )

        # Create output directory if it doesn't exist
        output_dir = "visualizations"
        os.makedirs(output_dir, exist_ok=True)

        # 1. Response time vs batch size for each concurrency level
        fig1 = px.line(
            stable_df,
            x="batch_size",
            y="avg_response_time",
            color="concurrency",
            markers=True,
            title="Response Time vs Batch Size by Concurrency Level",
            labels={
                "batch_size": "Batch Size",
                "avg_response_time": "Average Response Time (s)",
                "concurrency": "Concurrency",
            },
            hover_data=["throughput_images_per_sec"],
        )

        # Add tooltips via custom hovertext
        fig1.update_traces(hovertext=stable_df["hover_text"])

        # Adjust layout
        fig1.update_layout(
            xaxis_title="Batch Size",
            yaxis_title="Response Time (seconds)",
            legend_title="Concurrency",
            hovermode="closest",
        )

        # Save the figure
        fig1.write_html(os.path.join(output_dir, "response_time_vs_batch_size.html"))
        click.echo(f"Created visualization: {os.path.join(output_dir, 'response_time_vs_batch_size.html')}")

        # 2. Response time vs concurrency for each batch size
        # Pivot to get distinct batch size curves
        batch_sizes = stable_df["batch_size"].unique()  # noqa

        fig2 = px.line(
            stable_df,
            x="concurrency",
            y="avg_response_time",
            color="batch_size",
            markers=True,
            title="Response Time vs Concurrency by Batch Size",
            labels={
                "concurrency": "Concurrency",
                "avg_response_time": "Average Response Time (s)",
                "batch_size": "Batch Size",
            },
            hover_data=["throughput_images_per_sec"],
        )

        # Add tooltips via custom hovertext
        fig2.update_traces(hovertext=stable_df["hover_text"])

        # Adjust layout
        fig2.update_layout(
            xaxis_title="Concurrency",
            yaxis_title="Response Time (seconds)",
            legend_title="Batch Size",
            hovermode="closest",
        )

        # Save the figure
        fig2.write_html(os.path.join(output_dir, "response_time_vs_concurrency.html"))
        click.echo(f"Created visualization: {os.path.join(output_dir, 'response_time_vs_concurrency.html')}")

        # 3. Bonus visualization: Throughput heatmap
        # Create a pivot table for the heatmap
        pivot_df = stable_df.pivot_table(
            values="throughput_images_per_sec", index="concurrency", columns="batch_size", aggfunc="mean"
        )

        # Create the heatmap
        fig3 = go.Figure(
            data=go.Heatmap(
                z=pivot_df.values,
                x=pivot_df.columns,
                y=pivot_df.index,
                hoverongaps=False,
                colorscale="Viridis",
                colorbar=dict(title="Images/second"),
            )
        )

        fig3.update_layout(
            title="Throughput Heatmap (Images/second)",
            xaxis_title="Batch Size",
            yaxis_title="Concurrency",
        )

        # Save the heatmap
        fig3.write_html(os.path.join(output_dir, "throughput_heatmap.html"))
        click.echo(f"Created visualization: {os.path.join(output_dir, 'throughput_heatmap.html')}")

    except Exception as e:
        click.echo(f"Error generating visualizations: {e}")


if __name__ == "__main__":
    optimize_throughput()
