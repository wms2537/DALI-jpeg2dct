# Extract DCT Coefficients from JPEG Images with Nvidia DALI

Current implementation only supports cpu backend. Inspired by [this repository](https://github.com/uber-research/jpeg2dct).

## Prerequisites
DALI is installed from the binary distribution or compiled the from source.

## How to Use
Clone the repository and `cd` into it. Then
```bash
mkdir build && cd build
cmake ..
make -j4
```
After this, a file named `libjpeg2dct.so` can be found in your build folder.

Example
```python
import nvidia.dali.fn as fn
import nvidia.dali.plugin_manager as plugin_manager
plugin_manager.load_library('./build/libjpeg2dct.so')
help(fn.dct_to_jpeg)
```

Further example code can be found in [`test_op.py`](test_op.py).