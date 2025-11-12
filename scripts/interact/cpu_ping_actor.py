#!/usr/bin/env python3
import os
import ray
import click


@ray.remote  # CPU-only actor (no GPUs requested)
class PingActor:
    def __init__(self):
        self.count = 0

    def ping(self) -> str:
        return "pong"

    def inc(self, n: int = 1) -> int:
        self.count += n
        return self.count

    def get_count(self) -> int:
        return self.count


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli() -> None:
    """Manage a CPU-only PingActor on the Ray cluster."""
    pass


@cli.command()
@click.option(
    "--address",
    type=str,
    default=None,
    help="Ray address (e.g., ray://ray-head:10001). Defaults to RAY_ADDRESS or ray://ray-head:10001",
)
@click.option("--namespace", type=str, default="default", show_default=True, help="Ray namespace for the actor")
@click.option("--actor-name", type=str, default="cpu_ping", show_default=True, help="Detached actor name")
@click.option(
    "--run-sample", is_flag=True, default=False, show_default=True, help="Run a sample call sequence after deployment"
)
@click.option("--sample-inc", type=int, default=1, show_default=True, help="Increment amount for sample")
def deploy(address: str | None, namespace: str, actor_name: str, run_sample: bool, sample_inc: int) -> None:
    """Deploy the CPU-only actor and optionally run a sample query."""
    resolved = address or os.environ.get("RAY_ADDRESS", "ray://ray-head:10001")
    print(f"Connecting to Ray at: {resolved}")
    ray.init(address=resolved, namespace=namespace, ignore_reinit_error=True)

    # Get or create the detached actor
    try:
        actor = ray.get_actor(actor_name, namespace=namespace)
        print(f"Found existing actor: {actor_name}")
    except Exception:
        print(f"Creating actor: {actor_name}")
        actor = PingActor.options(name=actor_name, lifetime="detached").remote()

    if run_sample:
        print("Running sample sequence...")
        print("  ping     ->", ray.get(actor.ping.remote()))
        print(f"  inc({sample_inc}) ->", ray.get(actor.inc.remote(sample_inc)))
        print("  get_count->", ray.get(actor.get_count.remote()))


@cli.command()
@click.option(
    "--address",
    type=str,
    default=None,
    help="Ray address (e.g., ray://ray-head:10001). Defaults to RAY_ADDRESS or ray://ray-head:10001",
)
@click.option("--namespace", type=str, default="default", show_default=True, help="Ray namespace for the actor")
@click.option("--actor-name", type=str, default="cpu_ping", show_default=True)
@click.option("--action", type=click.Choice(["ping", "inc", "get_count"]), default="ping", show_default=True)
@click.option("--n", type=int, default=1, show_default=True, help="Increment amount when action=inc")
def query(address: str | None, namespace: str, actor_name: str, action: str, n: int) -> None:
    """Send a command to the deployed actor and print the response."""
    resolved = address or os.environ.get("RAY_ADDRESS", "ray://ray-head:10001")
    print(f"Connecting to Ray at: {resolved}")
    ray.init(address=resolved, namespace=namespace, ignore_reinit_error=True)
    actor = ray.get_actor(actor_name, namespace=namespace)

    if action == "ping":
        print(ray.get(actor.ping.remote()))
    elif action == "inc":
        print(ray.get(actor.inc.remote(n)))
    elif action == "get_count":
        print(ray.get(actor.get_count.remote()))


@cli.command()
@click.option(
    "--address",
    type=str,
    default=None,
    help="Ray address (e.g., ray://ray-head:10001). Defaults to RAY_ADDRESS or ray://ray-head:10001",
)
@click.option("--namespace", type=str, default="default", show_default=True, help="Ray namespace for the actor")
@click.option("--actor-name", type=str, default="cpu_ping", show_default=True)
def stop(address: str | None, namespace: str, actor_name: str) -> None:
    """Terminate the detached actor."""
    resolved = address or os.environ.get("RAY_ADDRESS", "ray://ray-head:10001")
    print(f"Connecting to Ray at: {resolved}")
    ray.init(address=resolved, namespace=namespace, ignore_reinit_error=True)
    actor = ray.get_actor(actor_name, namespace=namespace)
    ray.kill(actor)
    print(f"Killed actor: {actor_name}")


if __name__ == "__main__":
    cli()
