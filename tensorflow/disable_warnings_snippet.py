# Description:
# This disables everything except error messages displayed when using tensorflow.
# Code has to be added before anything else to work.

import warnings
warnings.filterwarnings('ignore')
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow.python.platform import tf_logging as logging
logging.set_verbosity(logging.ERROR)