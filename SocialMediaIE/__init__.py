import logging
import os
import sys
logging.getLogger(__name__).addHandler(logging.NullHandler())

_PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
def get_resource(relative_path):
    """Get absolute path to a resource delative to the package.

    relative_path: path relative to the package. e.g. data/file.txt
    """
    return os.path.join(_PROJECT_ROOT, relative_path)

