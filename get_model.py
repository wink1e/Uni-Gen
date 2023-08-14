import argparse
from unicore.data import Dictionary

from . import model


def define_model(text_path: str):
    parser = argparse.Namespace()
    parser.encoder = argparse.Namespace()
    parser.decoder = argparse.Namespace()

    dict_temp = Dictionary.load(text_path)
    models = model.UniMolGen(parser, dict_temp)
    return models
