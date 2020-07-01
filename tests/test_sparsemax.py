import torch
from torch.autograd import gradcheck, Variable
from torchsparseattn.sparsemax import sparsemax_function


def test_sparsemax():

    torch.manual_seed(1)
    torch.set_default_tensor_type("torch.DoubleTensor")

    for _ in range(30):
        func = sparsemax_function
        x = Variable(torch.randn(20), requires_grad=True)
        assert gradcheck(func, (x,), eps=1e-4, atol=1e-3)
