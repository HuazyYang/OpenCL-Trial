# OpenCL GPGPU Extensive Programing Examples

This repos demonstrates OpenCL GPGPU and its extensive programing skills, which include but not limited to following objectives:
 * Algebra solutions, such as matrix transpose(in-place or out-place), matrix multiplication, matrix-vector multiplication, tridiagonal system solvers(Thomas, CR, PCR, RD and their hybrids);
 * grayscale image histogram solution;
 * OpenCL-OpenGL interoperation examples: simulating heat transfer equation and visualizing the results, Depth of View Blur sample(Solving discreted heat diffusion equation and visualizing the results);
 * More examples will be coming soon.

# Build
 ## Prerequires
  * VC++ compiler support C++17 or higher, install Visual studio 2017 and above with Visual C++ workload is recommanded;
  * cmake with version 3.19.* or above;
  * OpenCL SDK with OpenCL Specification 2.0 or above installed: <br>[Intel OpenCL SDK](https://software.intel.com/content/www/us/en/develop/tools/opencl-sdk.html) is a common choise for desktop NVIDIA/AMD GPU, other SDKs from GPU venders may also be applicable(I have not tested them yet), [Khronos Group OpenCL SDK](https://github.com/KhronosGroup/OpenCL-SDK/) is also a alternative choise(Already tested).
  * Third party repos must be placed in the parent directory of this repos's local copy with subdirectory name "ThirdParty". Third party reposes must provide OpenGL development utlities:
    + [glad](https://glad.dav1d.de/), OpenCL Version 4.6 or above, Core Profile.
    +  [glm](https://github.com/g-truc/glm)
    + [glfw](https://github.com/glfw/glfw)
    + [spirv-1.1](https://github.com/KhronosGroup/SPIR/tree/spirv-1.1)
  * Optional development tools: Visual Studio Code with extensions: C/C++, CMake, CMake Tools, OpenCL, Output Colorizer(These is my preferred development evironment). Alternatively, you can use Visual Studio on windows if CMake support workload has been included in your installation.

 ## Build Steps
  Following standard cmake build procedure if you use VSCode(Check `Set CMAKE_BUILD_TYPE also on multi config generators` in `Settings`), or you need the following additionall settings(Again, I suggest you to use VSCode):
  * Before build debug version, define `CMAKE_BUILD_TYPE=Debug` in your cmake cache(CMakeCache.txt), and just kick cmake default build procedure;
  * Before build release version, select cmake variant to Release, then edit CMakeCache.txt with the option:`CMAKE_BUILD_TYPE=Release`, then kick off cmake build procedure.

 # Runtime Bugs Note
  I can only guarantee all the examples will be work properly for AMD GPU, as for NVIDIA GPU, the following known issues must be solved if you are trapped by a debug asset or objectionable wrong outputs:
   * Fill arguments corresponding to `local_work_size` in API calls `clEnqueueNDRangeKernel`;
   * In file `matrix/mat_mul_vec.cl`, synchronize all the local tile reduction writes operations;