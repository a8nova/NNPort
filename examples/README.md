This folder is a small demo project you can point NNPort at.

It’s meant to show NNPort iterating on a real OpenCL example until it runs correctly on the target device.

What to look at:
- `simple_opencl.cpp`: the host code NNPort edits/fixes during iteration
- `simple_opencl.cl`: the buggy kernel

Run NNPort in “Local path” mode and choose this `examples/` folder as the project.


