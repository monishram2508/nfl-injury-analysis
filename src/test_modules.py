# run this as a quick test, not in your src folder
import numpy as np
print(np.__version__)  # must print 1.26.4

from mmpose.apis import init_model, inference_topdown
from mmpose.utils import register_all_modules
register_all_modules()
print("mmpose imported cleanly")
