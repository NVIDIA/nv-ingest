import json
import yaml
import click
import os

from nv_ingest.api.main import app


@click.command()
@click.option("--output", default="openapi.yaml", help="Path to OpenAPI output file (default: openapi.json)")
def write_openapi_schema(output):
    if os.path.isdir(output):
        print(f"Warning: '{output}' is a directory. Defaulting to '{output}/openapi.yaml'.")
        output = os.path.join(output, "openapi.yaml")

    # Determine format based on file extension
    if output.endswith(".yaml") or output.endswith(".yml"):
        with open(output, "w") as f:
            yaml.dump(app.openapi(), f, default_flow_style=False)
        print(f"OpenAPI YAML written to: {output}")
    else:
        with open(output, "w") as f:
            json.dump(app.openapi(), f, indent=4)
        print(f"OpenAPI JSON written to: {output}")


if __name__ == "__main__":
    write_openapi_schema()
