"""vccpipe: the pipeline behind our Virtual Cell Challenge 2026 submission.

`from vccpipe import *` gives the notebook the package's public functions, classes and constants (each module lists
them in its `__all__`), plus `show`. The modules, from the inputs to the final file:

- `inputs`: the public screens as tables; each knockdown's borrowed profile, the significance gate, the neighbour prior.
- `constants`: numbers fitted once on the training screens (the per-cell amplitude table, the neighbour prior).
- `levels`: the rank test's power, measured from a line's own controls.
- `combiner`: E, the learned combiner, a small ridge regression across cell types.
- `generator`: stage 2, from a predicted change to 400 cells of whole-number counts.
- `stage1`: stage 1's pieces, from the borrowed profiles to each knockdown's targets.
- `phase`: the entry points: read the challenge's input, prepare the references, run the probe, build the file.
- `harness`: a training line held out like the challenge and scored by the official scorer.
- `figures`: the palette and the helpers the figures share.
- `rebuild`: rebuilding the reference tables from the public downloads (the notebook's appendix).
- `fast`: FAST mode's small inputs, read from the public files by byte range.
"""

from . import constants, generator, inputs, levels
from .constants import *
from .generator import *
from .inputs import *
from .levels import *

_MODULES = (inputs, constants, levels, generator,)
__all__ = [name for module in _MODULES for name in module.__all__] + ["show"]


def show(obj):
    """Display the source code of a function or class in the notebook.

    The notebook's sections call it to show the code they explain.
    """
    import inspect

    from IPython.display import Code, display

    display(Code(inspect.getsource(obj), language="python"))
