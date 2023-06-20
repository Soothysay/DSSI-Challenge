from .squeezenet import SqueezeNet

def get_network(network_name):
    if network_name == 'squeezenet':
        network = SqueezeNet()
    else:
        raise NotImplementedError
    return network