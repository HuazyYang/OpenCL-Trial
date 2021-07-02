# D3D11 Memory Model

The following table decribles how the resource will need to be accesssed
by CPU and/or by GPU, there will be performance tradeoffs.

| CPU/GPU r/w | Default | Dynamic | Immutable | Staging |
| ---- | ---- | ---- | ---- | ---- |
| GPU-Read|  yes | yes | yes | yes |
| GPU-Write | yes | no | no | yes |
| CPU read | no | no | no | yes |
| CPU write | yes | yes | no | yes |

For D3D11, there are the usage enumeration
```
typedef enum D3D11_USAGE {
  D3D11_USAGE_DEFAULT,
  D3D11_USAGE_IMMUTABLE,
  D3D11_USAGE_DYNAMIC,
  D3D11_USAGE_STAGING
} ;
```
Check D3D11 document of [`D3D11_USAGE`](https://docs.microsoft.com/en-us/windows/win32/api/d3d11/ne-d3d11-d3d11_usage) for details.


# D3D12/Vulan Memory Model
D3D12 heap type:
```
typedef enum D3D12_HEAP_TYPE {
  D3D12_HEAP_TYPE_DEFAULT,
  D3D12_HEAP_TYPE_UPLOAD,
  D3D12_HEAP_TYPE_READBACK,
  D3D12_HEAP_TYPE_CUSTOM
} ;
```
Which correspond to `Vulkan` memory types:
  - `DEVICE_LOCAL` / `D3D12_HEAP_TYPE_DEFAULT`
    + **Video memory**. Fast access from GPU;
    + No direct access from CPU – mapping not possible;
    + Pros/Cons:
      * Good for resources written and read frequently by GPU;
      * Good for resources uploaded once (immutable) or infrequently by CPU, read frequently by GPU.
  - `HOST_VISIBLE` / `D3D12_HEAP_TYPE_UPLOAD`
    + **System memory** Accessible to CPU – mapping possible;
    + ***Uncached***. Writes may be write-combined;
    + Access from GPU possible but slow Across PCIe® bus, reads cached on GPU.
    + Pros/Cons:
      * Good for CPU-side (staging) copy of your resources – used as source of transfer.
      * Data written by CPU, read once by GPU (e.g. constant buffer) may work fine (always measure!) <br>
       *Cache on GPU may help*.
      * Large data read by GPU – place here as *last resort*.
      * Large data written and read by GPU – *shouldn’t* ever be here.
  - `DEVICE_LOCAL+HOST_VISIBLE`
    + **Special pool of video memory**.
    + Exposed on AMD only. 256 MiB.
    + Fast access from GPU.
    + Accessible to CPU – mapping possible.
      * Written directly to video memory.
      * Writes may be write-combined.
      * Uncached. Don’t read from it.
    + Pros/Cons:
      * Good for resources updated frequently by CPU (dynamic), read by GPU.
      * Direct access by both CPU and GPU – you don’t need to do explicit transfer.
      * Use as fallback if DEVICE_LOCAL is small and oversubscribed;

  - `HOST_VISIBLE+HOST_CACHED` / `D3D12_HEAP_TYPE_READBACK`
    + **System memory**;
    + CPU reads and writes cached (write-back);
    + GPU access through PCIe;
      * GPU reads snoop CPU cache.
    + Pros/Cons:
      * Good for resources written by GPU, read by CPU – results of computations;
      * Direct access by both CPU and GPU – you don’t need to do explicit transfer;
      * Use for any resources read or accessed randomly on CPU.

# Memory types: AMD APU
  ...

# Memory Management Tips and Tricks
  ## Suballocation
  - Don’t allocate separate memory block for each resource (DX12: CreateCommittedResource).
    + small limit on maximum number of allocations (e.g. 4096)
    + allocation is slow
  - Prefer not to allocate or free memory blocks during gameplay to avoid hitching. <br>
    *If you need to, you can do it on background thread.*
  - Allocate bigger blocks and sub-allocate ranges for your resources (DX12: CreatePlacedResource).
    + 256 MiB is good default block size
    + For heaps <= 1 GiB use smaller blocks (e.g. heap size / 8).

  ## Over-Commitment
  *What happens when you exceed the maximum amount of physical video memory?*
  - It depends on the driver.
    + Allocation may fail (VK_ERROR_OUT_OF_DEVICE_MEMORY).
    + Allocation may succeed (VK_SUCCESS), e.q. Some blocks are silently migrated to system memory.
  - Blocks may be migrated to system memory anyway
    + You are not alone – other applications can use video memory.
    + Using blocks migrated to system memory on GPU degrades performance.

  ## Over-Commitment--Vulkan™
  Refer to [Memory management in Vulkan and DX12](https://ubm-twvideo01.s3.amazonaws.com/o1/vault/gdc2018/presentations/Sawicki_Adam_Memory%20management%20in%20Vulkan.pdf).

  ## Over-Commitment--DX12
  - Size of memory types: <br>
    `DXGI_ADAPTER_DESC`
  - Current usage and available budget for your program: <br>
    `DXGI_QUERY_VIDEO_MEMORY_INFO`
  - You can register for notifications: <br>
    `IDXGIAdapter3::RegisterVideoMemoryBudgetChangeNotificationEvent`
  - You can page allocated blocks (heaps) in and out of video memory: <br>
    `ID3D12Device::Evict`, `MakeResident`, `ID3D12Device3::EnqueueMakeResident`
  - You can set residency priorities to resources: <br>
    `ID3D12Device1::SetResidencyPriority`
  -  You can inform DX12 about minimum required memory: <br>
    `IDXGIAdapter3::SetVideoMemoryReservation`

  ## Mapping -- `void *ptr;`
  - You can inform DX12 about minimum required memory: <br>
    `IDXGIAdapter3::SetVideoMemoryReservation` <br>
    *You don’t need to unmap before using on GPU*. OK for `HOST_VISIBLE`/`D3D12_HEAP_TYPE_UPLOAD`.
  - Exceptions:
    + Vulkan™, AMD, Windows® version < 10: Blocks of
      `DEVICE_LOCAL + HOST_VISIBLE`/`READBACK` memory that stay mapped for the time of any call to Submit or Present are migrated to system memory.
    + Keeping many large memory blocks mapped may impact stability or performance of **debugging tools** -- another reason for suballocation of resource type `HOST_VISIBLE`/`D3D12_HEAP_TYPE_UPLOAD`.

  ## Transfer -- CPU Buffer to GPU heap and CPU Readback
  - Copy queue is designed for efficient transfer via **PCIe**
    + Use it in parallel with 3D rendering, even asynchronously to rendering frames. Good for *texture streaming*.
    + Use it also for defragmentation of GPU memory in the background.
    + Do your transfers long before the data is needed on graphics queue.
  - GPU to (the same) GPU copies are much faster on  graphics queue.
    + Use it if graphics queue needs to wait for transfer result anyway.








