import pytest
import torch


def pytest_addoption(parser):
    parser.addoption(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "all"],
        help="device for test",
    )


def pytest_generate_tests(metafunc):
    if "device" in metafunc.fixturenames:
        specified_device = metafunc.config.getoption("device")

        if specified_device == "all":
            metafunc.parametrize(
                "device", [torch.device("cpu"), torch.device("cuda")])
        else:
            metafunc.parametrize("device", [torch.device(specified_device)])
