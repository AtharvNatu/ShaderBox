- This folder contains the demonstration code for Vulkan-CUDA Interoperability demonstrated using Sine Wave Computation

- This application uses vkCmdDrawIndirect() instead of vkCmdDraw() to switch between mesh sizes by using Indirect Buffer.

- To switch between CPU and GPU on key press, uniform buffer is used and appropriate input attribute is chosen in the vertex shader.

- This code does not contain Semaphore based synchronization between CUDA and Vulkan and uses rudimentary call to cudaDeviceSynchronize() as placeholder for the next version to add semaphores.

- Additionally, this application allows to increase mesh size by clicking following keys (with associated mesh sizes)
    1 - 64
    2 - 128
    3 - 256
    4 - 512
    5 - 1024
    6 - 2048
    7 - 4096

- To Change Sine wave color, use UP arrow key and rotate between different colors - Orange, Red, Green, Blue, Cyan, Magenta, Yellow and White