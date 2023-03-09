import os
import torch


def setup_device(app, use_cuda=True):
    """
    Initialize the torch device where the code will be executed on.

    :param app: the app, to format logging
    :param use_cuda: Set to True if you want the code to be run on your GPU. If set to False, code will run on CPU.
    :return: torch.device : The initialized device, torch.device.
    """
    if use_cuda is False or not torch.cuda.is_available():
        device_name = "cpu"
        if use_cuda is True:
            app.logger.critical("unable to initialize CUDA, check torch installation (https://pytorch.org/)")
        if use_cuda is False:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        device_name = "cuda:0"
        app.logger.info("CUDA successfully initialized on device : " + torch.cuda.get_device_name())

    device = torch.device(device_name)

    app.logger.info("Using device : " + device.type)

    return device
