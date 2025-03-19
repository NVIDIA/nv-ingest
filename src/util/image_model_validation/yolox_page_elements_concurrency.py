import base64
import json
import time
import requests
import click
from concurrent.futures import ThreadPoolExecutor, as_completed


@click.command()
@click.option(
    "--image-path", required=True, type=click.Path(exists=True), help="Path to the image file to be encoded and sent."
)
@click.option("--url", default="http://localhost:8000/v1/infer", help="Endpoint URL to send the POST request.")
@click.option("--start", default=1, help="Starting number of concurrent calls.", type=int)
@click.option("--max-concurrency", default=1024, help="Maximum number of concurrent calls to test.", type=int)
@click.option("--timeout", default=5, help="Timeout (in seconds) for each request.", type=int)
@click.option(
    "--start-batch", default=2, help="Starting number of times to include the image in each payload.", type=int
)
@click.option("--max-batch", default=16, help="Maximum number of times to include the image in each payload.", type=int)
@click.option("--auth-token", default=None, help="Authentication token for the Authorization header (Bearer token).")
@click.option(
    "--custom-headers", default="", help='Additional headers as a JSON string, e.g. \'{"X-Custom": "value"}\'.'
)
def test_concurrency(
    image_path, url, start, max_concurrency, timeout, start_batch, max_batch, auth_token, custom_headers
):
    """
    Tests the YOLOX page elements service with varying concurrency and batch sizes.

    This script performs two tests:
    1. Batch size scaling: Increases batch size while keeping concurrency constant
    2. Concurrency scaling: Increases concurrent calls while keeping batch size constant

    Both tests analyze performance and identify limits of the service.
    """
    # Read and encode the image file once (store the raw base64 string)
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

    # Set up default headers and update with auth token or custom headers if provided
    headers = {"accept": "application/json", "content-type": "application/json"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    if custom_headers:
        try:
            extra_headers = json.loads(custom_headers)
            headers.update(extra_headers)
        except Exception as e:
            click.echo(f"Error parsing custom headers: {e}. Please provide a valid JSON string.")
            return

    throttling_logs = []
    all_results = {}

    # PART 1: Batch Size Scaling Test
    current_concurrency = start
    current_batch = start_batch

    click.echo(f"\n=== Testing batch size scaling with fixed concurrency {current_concurrency} ===")
    while current_batch <= max_batch:
        click.echo(f"\nTesting with batch size of {current_batch}...")
        start_time = time.time()
        results = []

        # Launch concurrent requests using a thread pool
        with ThreadPoolExecutor(max_workers=current_concurrency) as executor:
            futures = [
                executor.submit(send_yolox_request, url, encoded_image, current_batch, timeout, headers)
                for _ in range(current_concurrency)
            ]
            for future in as_completed(futures):
                results.append(future.result())

        elapsed = time.time() - start_time

        # Calculate success rate
        success_count = sum(1 for r in results if isinstance(r, int) and r >= 200 and r < 500)
        success_rate = (success_count / current_concurrency) * 100 if current_concurrency > 0 else 0

        click.echo(f"Elapsed time for batch size {current_batch}: {elapsed:.2f} seconds")
        click.echo(f"Success rate: {success_rate:.2f}%")

        # Tally response types
        timeouts = results.count("timeout")
        errors = results.count("error")
        code_500 = results.count(500)
        code_429 = results.count(429)

        click.echo(
            f"Response summary: 200s: {results.count(200)}, 429s: {code_429}, 500s: {code_500},"
            f"timeouts: {timeouts}, errors: {errors}"
        )

        # Store results for this batch size
        all_results[f"batch_{current_batch}_concurrency_{current_concurrency}"] = {
            "elapsed_time": elapsed,
            "success_rate": success_rate,
            "results": {
                "200s": results.count(200),
                "429s": code_429,
                "500s": code_500,
                "timeouts": timeouts,
                "errors": errors,
            },
        }

        # Stop batch testing if any timeout, error, or 500 is encountered
        if timeouts > 0 or errors > 0 or code_500 > 0:
            click.echo(f"Stopping batch size test: encountered errors at batch size {current_batch}.")
            break

        # Increase batch size (doubling for this example)
        current_batch *= 2

    # Store the optimal batch size for the concurrency test
    optimal_batch = max(start_batch, current_batch // 2)
    if timeouts > 0 or errors > 0 or code_500 > 0 and current_batch > start_batch:
        optimal_batch = current_batch // 2

    # PART 2: Concurrency Scaling Test
    click.echo(f"\n=== Testing concurrency scaling with fixed batch size {optimal_batch} ===")
    current_concurrency = start
    while current_concurrency <= max_concurrency:
        click.echo(f"\nTesting with {current_concurrency} concurrent calls...")
        start_time = time.time()
        results = []

        # Launch concurrent requests using a thread pool
        with ThreadPoolExecutor(max_workers=current_concurrency) as executor:
            futures = [
                executor.submit(send_yolox_request, url, encoded_image, optimal_batch, timeout, headers)
                for _ in range(current_concurrency)
            ]
            for future in as_completed(futures):
                results.append(future.result())

        elapsed = time.time() - start_time

        # Calculate success rate
        success_count = sum(1 for r in results if isinstance(r, int) and r >= 200 and r < 500)
        success_rate = (success_count / current_concurrency) * 100 if current_concurrency > 0 else 0

        click.echo(f"Elapsed time for {current_concurrency} concurrent calls: {elapsed:.2f} seconds")
        click.echo(f"Success rate: {success_rate:.2f}%")

        # Tally response types
        timeouts = results.count("timeout")
        errors = results.count("error")
        code_500 = results.count(500)
        code_429 = results.count(429)

        click.echo(
            f"Response summary: 200s: {results.count(200)}, 429s: {code_429}, 500s: {code_500}, "
            f"timeouts: {timeouts}, errors: {errors}"
        )

        if code_429:
            throttling_logs.append((current_concurrency, code_429))
            click.echo(f"Throttling: {code_429} responses with 429 status at {current_concurrency} concurrent calls.")

        # Store results for this concurrency level
        all_results[f"batch_{optimal_batch}_concurrency_{current_concurrency}"] = {
            "elapsed_time": elapsed,
            "success_rate": success_rate,
            "results": {
                "200s": results.count(200),
                "429s": code_429,
                "500s": code_500,
                "timeouts": timeouts,
                "errors": errors,
            },
        }

        # Stop if any timeout, error, or 500 is encountered
        if timeouts > 0 or errors > 0 or code_500 > 0:
            click.echo(
                f"Stopping test: encountered {timeouts} timeouts, {errors} errors, "
                f"{code_500} 500s at {current_concurrency} concurrent calls."
            )
            break

        # Increase concurrency (doubling for this example)
        current_concurrency *= 2

    # Generate summary and recommendations
    click.echo("\n=== Test Summary ===")
    max_successful_batch = optimal_batch
    max_successful_concurrency = max(start, current_concurrency // 2) if current_concurrency > start else start

    click.echo(f"Maximum successful batch size: {max_successful_batch}")
    click.echo(f"Maximum successful concurrency: {max_successful_concurrency}")

    # Calculate optimal throughput configuration
    click.echo("\n=== Recommendations ===")
    click.echo("Recommended configuration for optimal throughput:")
    click.echo(f"- Batch size: {max_successful_batch}")
    click.echo(f"- Concurrency: {max_successful_concurrency}")

    # Save all results to a file
    save_results_to_file(all_results)

    # Report throttling events if any were logged
    if throttling_logs:
        click.echo("\nThrottling Log (429 responses):")
        for concurrency, count in throttling_logs:
            click.echo(f"Concurrent calls: {concurrency} -> 429 responses: {count}")

    # PART 3: Sustained Load Test at Optimal Settings
    click.echo("\n=== Starting Sustained Load Test at Optimal Settings ===")
    click.echo(f"Running with batch size {max_successful_batch} and concurrency {max_successful_concurrency}")
    click.echo("Press Ctrl+C to stop the test")

    sustained_results = {
        "total_requests": 0,
        "successful_requests": 0,
        "failed_requests": 0,
        "avg_response_time": 0,
        "start_time": time.time(),
        "intervals": [],
    }

    interval_count = 0
    interval_duration = 120  # seconds
    last_report_time = time.time()

    try:
        while True:
            interval_start_time = time.time()
            _ = interval_start_time
            interval_results = []

            # Launch concurrent requests using a thread pool
            with ThreadPoolExecutor(max_workers=max_successful_concurrency) as executor:
                futures = [
                    executor.submit(
                        send_yolox_request_with_timing, url, encoded_image, max_successful_batch, timeout, headers
                    )
                    for _ in range(max_successful_concurrency)
                ]
                for future in as_completed(futures):
                    status, response_time = future.result()
                    interval_results.append((status, response_time))

            # Process interval results
            success_count = sum(
                1 for status, _ in interval_results if isinstance(status, int) and status >= 200 and status < 500
            )
            total_count = len(interval_results)

            # Update sustained results
            sustained_results["total_requests"] += total_count
            sustained_results["successful_requests"] += success_count
            sustained_results["failed_requests"] += total_count - success_count

            # Calculate average response time for successful requests in this interval
            successful_times = [
                t for status, t in interval_results if isinstance(status, int) and status >= 200 and status < 500
            ]
            avg_time = sum(successful_times) / len(successful_times) if successful_times else 0

            # Update rolling average response time
            if sustained_results["avg_response_time"] == 0:
                sustained_results["avg_response_time"] = avg_time
            else:
                sustained_results["avg_response_time"] = (
                    sustained_results["avg_response_time"] * interval_count + avg_time
                ) / (interval_count + 1)

            # Store interval data
            interval_data = {
                "interval": interval_count,
                "requests": total_count,
                "successes": success_count,
                "failures": total_count - success_count,
                "avg_response_time": avg_time,
                "response_codes": {
                    "200": sum(1 for status, _ in interval_results if status == 200),
                    "429": sum(1 for status, _ in interval_results if status == 429),
                    "500": sum(1 for status, _ in interval_results if status == 500),
                    "timeout": sum(1 for status, _ in interval_results if status == "timeout"),
                    "error": sum(1 for status, _ in interval_results if status == "error"),
                },
            }
            sustained_results["intervals"].append(interval_data)

            # Report every 10 seconds
            current_time = time.time()
            if current_time - last_report_time >= interval_duration:
                elapsed_total = current_time - sustained_results["start_time"]
                success_rate = (
                    (sustained_results["successful_requests"] / sustained_results["total_requests"]) * 100
                    if sustained_results["total_requests"] > 0
                    else 0
                )
                requests_per_second = sustained_results["total_requests"] / elapsed_total if elapsed_total > 0 else 0

                click.echo(f"\nInterval {interval_count} summary:")
                click.echo(f"Success rate: {success_rate:.2f}%")
                click.echo(f"Requests per second: {requests_per_second:.2f}")
                click.echo(f"Average response time: {sustained_results['avg_response_time']:.4f} seconds")
                click.echo(f"Total requests: {sustained_results['total_requests']}")

                # Save progress periodically
                save_results_to_file(sustained_results, filename="yolox_sustained_results.json")
                last_report_time = current_time

            interval_count += 1

    except KeyboardInterrupt:
        click.echo("\n\n=== Sustained Load Test Stopped ===")
        elapsed_total = time.time() - sustained_results["start_time"]
        success_rate = (
            (sustained_results["successful_requests"] / sustained_results["total_requests"]) * 100
            if sustained_results["total_requests"] > 0
            else 0
        )
        requests_per_second = sustained_results["total_requests"] / elapsed_total if elapsed_total > 0 else 0

        click.echo("\nFinal Results:")
        click.echo(f"Total run time: {elapsed_total:.2f} seconds")
        click.echo(f"Total requests: {sustained_results['total_requests']}")
        click.echo(f"Successful requests: {sustained_results['successful_requests']}")
        click.echo(f"Failed requests: {sustained_results['failed_requests']}")
        click.echo(f"Success rate: {success_rate:.2f}%")
        click.echo(f"Requests per second: {requests_per_second:.2f}")
        click.echo(f"Average response time: {sustained_results['avg_response_time']:.4f} seconds")

        # Save final results
        save_results_to_file(sustained_results, filename="yolox_sustained_results.json")
        click.echo("Final results saved to yolox_sustained_results.json")


def send_yolox_request(url, encoded_image, batch_size, timeout, headers):
    """
    Constructs the payload using the provided base64 image data and sends a POST request.

    Payload construction follows the YOLOX page elements service format:
      - The image data is prefixed with "data:image/png;base64,".
      - Each payload contains an "input" key whose value is a list of image objects.

    Returns:
        - The status code if the request is successful.
        - "timeout" if the request times out.
        - "error" for any other exception.
    """
    # Build the input list by repeating the image (with prefix) for the specified batch size
    input_list = []
    for _ in range(batch_size):
        image_url = f"data:image/png;base64,{encoded_image}"
        image_obj = {"type": "image_url", "url": image_url}
        input_list.append(image_obj)

    # Construct the payload with the "input" key
    payload = {"input": input_list}

    try:
        response = requests.post(url, json=payload, timeout=timeout, headers=headers)
        return response.status_code
    except requests.exceptions.Timeout:
        return "timeout"
    except Exception as e:
        return e


def send_yolox_request_with_timing(url, encoded_image, batch_size, timeout, headers):
    """
    Similar to send_yolox_request but also returns the response time.

    Returns:
        - A tuple of (status_code, response_time) where:
          - status_code is the HTTP status code, "timeout", or "error"
          - response_time is the time taken for the request in seconds
    """
    input_list = []
    for _ in range(batch_size):
        image_url = f"data:image/png;base64,{encoded_image}"
        image_obj = {"type": "image_url", "url": image_url}
        input_list.append(image_obj)

    payload = {"input": input_list}

    start_time = time.time()
    try:
        response = requests.post(url, json=payload, timeout=timeout, headers=headers)
        response_time = time.time() - start_time
        return response.status_code, response_time
    except requests.exceptions.Timeout:
        response_time = time.time() - start_time
        return "timeout", response_time
    except Exception:
        response_time = time.time() - start_time
        return "error", response_time


def save_results_to_file(results, filename="yolox_concurrency_results.json"):
    """
    Saves the test results to a JSON file.
    """
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)

    click.echo(f"Results saved to {filename}")


if __name__ == "__main__":
    test_concurrency()
