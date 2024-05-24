"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.
"""
import importlib
import logging
import os

logger = logging.getLogger(__name__)

EVALUATOR_REGISTRY = {}


def register_evaluator(name, subname=None):
    """
    New model types can be added to fairseq with the :func:`register_model`
    function decorator.

    For example::

        @register_model('lstm')
        class LSTM(FairseqEncoderDecoderModel):
            (...)

    .. note:: All models must implement the :class:`BaseFairseqModel` interface.
        Typically you will extend :class:`FairseqEncoderDecoderModel` for
        sequence-to-sequence tasks or :class:`FairseqLanguageModel` for
        language modeling tasks.

    Args:
        name (str): the name of the model
        :param name:
        :param subname:
    """

    def register_evaluator_cls(cls):
        if subname is None:
            if name in EVALUATOR_REGISTRY:
                raise ValueError('Cannot register duplicate model ({})'.format(name))
            EVALUATOR_REGISTRY[name] = cls
        else:
            if name in EVALUATOR_REGISTRY and subname in EVALUATOR_REGISTRY[name]:
                raise ValueError('Cannot register duplicate model ({}/{})'.format(name, subname))
            EVALUATOR_REGISTRY.setdefault(name, {})
            EVALUATOR_REGISTRY[name][subname] = cls
        return cls

    return register_evaluator_cls


# automatically import any Python files in the models/ directory
datasets_dir = os.path.dirname(__file__)
for file in os.listdir(datasets_dir):
    path = os.path.join(datasets_dir, file)
    if (
            not file.startswith('_')
            and not file.startswith('.')
            and (file.endswith('.py') or os.path.isdir(path))
    ):
        model_name = file[:file.find('.py')] if file.endswith('.py') else file
        module = importlib.import_module(f'wassa.evaluators.{model_name}')
