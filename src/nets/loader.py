from .qualityNet import QualityNet


def load_model(path):
    net = QualityNet()
    net.loading(path).eval()
    return net
    