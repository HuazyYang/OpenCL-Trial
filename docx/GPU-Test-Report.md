# 显卡测试性能综述
## 基准显卡和驱动程序
 AMD Radeon RX550/550 Series, 架构参数:
* GCN 4代架构
* CUs: 8, Cores: Cores
* VRAM: 4GB GDDR5.
* L1缓存: 16KB
* 可使用的局部存储容量: 32KB(总64KB)

驱动参数:
* Radeon Adrenalin 2020.21.3.1
* 版本: 27.20.15003.1004
## VMware虚拟显卡
NVIDIA Data Center/Telsa, M-Class Series, GRID M60-2Q, 架构参数:
* Tesla架构
* Cuda Cores: 2048.
* SM: 16, (128 Cores Per SM).
* VRAM: 1~2GB GDDR5, 内存接口总线宽度: 256-bits
* 可用共享内存: 8GB
驱动参数:
* 版本: 462.31, 支持DirectX12 Feature Level 12_1.

## 测试实例对比结果:
* 对D3D12和Vulkan的图形程序， 显存1G环境下，前向渲染(计算密集型应用)GPU管线提交执行效率NV卡比A卡帧率提高2.5x;
* 对D3D12 GP/GPU通用计算应用, 采用D3D Compute Shader模拟通用的高访存/计算比程序, N卡比A卡帧率提升在5.x以上.　由于驱动问题, 无法使用OpenCL/Cuda程序进行测试，但对Compute Shader的测试可直接反映出其性能差异.
