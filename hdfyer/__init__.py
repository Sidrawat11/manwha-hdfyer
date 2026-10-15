"""HDFyer: upscales manhwa chapters (CBZ) to 2x with a small distilled model.

  ingest/   find the chapters to do, read and decode them ahead of the GPU
  digest/   cut pages into strips, run the model (TensorRT or PyTorch), stitch the strips back
  render/   sharpen, clean up, encode and write the output CBZ
  pipeline  one chapter at a time; parallel: several chapters at once in worker processes
"""

import os

# Must be set before torch is imported anywhere: lets the allocator grow segments instead of fragmenting.
# Here rather than in main.py so --parallel worker processes (which import hdfyer, not main) get it too.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
