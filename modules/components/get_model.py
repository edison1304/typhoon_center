from utils.imports import *
from modules.models.center_finder import CenterFinder
from modules.models.Vit import TransformerRegressor


def get_model(config):
    if config['model'] == 'center_finder':
        return CenterFinder(in_channels=4)
    elif config['model'] == 'ViT':
        return TransformerRegressor(in_chans=4)
    else:
        raise ValueError(f"Invalid model name: {config['model']}")  
    