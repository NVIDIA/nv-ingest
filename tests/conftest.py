import random


def pytest_addoption(parser):
    parser.addoption(
        "--random-selection",
        metavar="N",
        action="store",
        default=-1,
        type=int,
        help="Only run random selected subset of N tests.",
    )


def pytest_collection_modifyitems(session, config, items):
    random_sample_size = config.getoption("--random-selection")

    if random_sample_size >= 0:
        items[:] = random.sample(items, k=random_sample_size)
