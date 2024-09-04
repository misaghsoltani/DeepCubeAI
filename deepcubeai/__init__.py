import re
import warnings

__version__ = "0.1.2"
__author__ = "Misagh Soltani"

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=re.escape("You are using `torch.load` with `weights_only=False` (the current default "
                      "value), which uses the default pickle module implicitly."))
