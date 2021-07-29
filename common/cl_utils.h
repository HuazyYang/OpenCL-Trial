#pragma once

#include <CL/cl.h>
#include <assert.h>
#include <type_traits>

#pragma once

typedef cl_int CLHRESULT;

#ifndef CL_SUCCEEDED
#define CL_SUCCEEDED(hr) ((hr) == CL_SUCCESS)
#endif /*CL_SUCCEEDED*/

#ifndef CL_FAILED
#define CL_FAILED(hr) ((hr) != CL_SUCCESS)
#endif /*CL_FAILED*/

#define CL_TRACE(...)  CLUtilsTrace(__VA_ARGS__)

void CLUtilsTrace(CLHRESULT hr, const char *fmt, ...);

#if defined(DEBUG) || defined(_DEBUG)
#ifndef V
#define V(x)           { hr = (x); if(hr != CL_SUCCESS) { CL_TRACE(hr, #x), assert( 0 ); } }
#endif
#ifndef V_RETURN
#define V_RETURN(x)    { hr = (x); if( hr != CL_SUCCESS ) { CL_TRACE(hr, #x), assert( 0 ); return hr; } }
#define V_RETURN2(x,...) { hr = (x,##__VA_ARGS__);  if( hr != CL_SUCCESS ) { CL_TRACE(hr, #x), assert( 0 ); return hr; } }
#endif
#else
#ifndef V
#define V(x)           { hr = (x); }
#endif
#ifndef V_RETURN
#define V_RETURN(x)    { hr = (x); if( hr != CL_SUCCESS ) { return hr; } }
#define V_RETURN2(x,...) { hr = (x,##__VA_ARGS__);  if( hr != CL_SUCCESS ) { return hr; } }
#endif
#endif

#ifndef SAFE_DELETE
#define SAFE_DELETE(p)       { if (p) { delete (p);     (p) = nullptr; } }
#endif
#ifndef SAFE_DELETE_ARRAY
#define SAFE_DELETE_ARRAY(p) { if (p) { delete[] (p);   (p) = nullptr; } }
#endif

#define RT_ASSERT   assert

inline ptrdiff_t AlignUp(ptrdiff_t ptr, size_t alignment) {
  size_t mask = alignment - 1;
  RT_ASSERT((alignment & mask) == 0 && "alignment must be pow of 2");
  if ((alignment & mask) != 0)
    ptr = (ptr + mask) & ~mask;
  return ptr;
}

inline ptrdiff_t AlignDown(ptrdiff_t ptr, size_t alignment) {
  size_t mask = alignment - 1;
  RT_ASSERT((alignment & mask) == 0 && "alignment must be pow of 2");
  if ((alignment & mask) != 0)
    ptr = ptr & ~mask;
  return ptr;
}

inline size_t RoundC(size_t c, size_t alignment) {
  size_t mask = alignment - 1;
  RT_ASSERT((alignment & mask) == 0 && "alignment must be pow of 2");
  if ((c & mask) != 0)
    c = (c + mask) & ~mask;
  return c;
}
inline size_t RoundF(size_t c, size_t alignment) {
  size_t mask = alignment - 1;
  RT_ASSERT((alignment & mask) == 0 && "alignment must be pow of 2");
  if ((c & mask) != 0)
    c = c & ~mask;
  return c;
}

enum class CLX_OBJECT_TYPE {
  PLATFORM_ID,
  DEVICE_ID,
  CONTEXT,
  PROGRAM,
  KERNEL,
  COMMAND_QUEUE,
  EVENT,
  MEM_OBJECT,
  BUFFER,
  IMAGE,
  SAMPLER,
  CLX_OBJECT_TYPE_MAX
};

typedef cl_int CLX_REFRET_T;

typedef CLX_REFRET_T(CL_API_CALL *CLX_OBJECT_ADDREF)(void *);
typedef CLX_REFRET_T(CL_API_CALL *CLX_OBJECT_RELEASE)(void *);

struct CLX_OBJECT_REFCOUNT_MGR_TABLE_ENTRY {
 CLX_OBJECT_ADDREF AddRef;
 CLX_OBJECT_RELEASE Release;
};

extern CLX_OBJECT_REFCOUNT_MGR_TABLE_ENTRY g_CLxObjectRefcountMgrTable[(int)CLX_OBJECT_TYPE::CLX_OBJECT_TYPE_MAX];

template<class ObjType, CLX_OBJECT_TYPE TypeIndex>
class CLxObjectSPtr {
public:
  static CLxObjectSPtr Move(ObjType p) {
    CLxObjectSPtr ptr;
    ptr.ptr_ = p;
    return ptr;
  }
  CLX_REFRET_T Attach(ObjType p) {
    CLX_REFRET_T ref = InternalRelease(ptr_);
    ptr_ = p;
    return ref;
  }
  CLxObjectSPtr() {
    ptr_ = nullptr;
  }
  CLxObjectSPtr(ObjType p) {
    InternalAddRef(ptr);
    ptr_ = p;
  }
  CLxObjectSPtr(const CLxObjectSPtr &p) {
    ptr_ = nullptr;
    InternalAddRef(p.ptr_);
    ptr_ = p.ptr_;
  }
  CLxObjectSPtr(CLxObjectSPtr &&p) {
    ptr_ = p.ptr_;
    p.ptr_ = nullptr;
  }
  ~CLxObjectSPtr() {
    InternalRelease(ptr_);
  }
  CLxObjectSPtr& operator = (ObjType p) {
    InternalAddRef(p);
    InternalRelease(ptr_);
    ptr_ = p;
    return *this;
  }
  CLxObjectSPtr& operator = (const CLxObjectSPtr &p) {
    InternalAddRef(p.ptr_);
    InternalRelease(ptr_);
    ptr_ = p.ptr_;
    return *this;
  }
  CLxObjectSPtr operator = (CLxObjectSPtr &&p) {
    InternalRelease(ptr_);
    ptr_ = p.ptr_;
    p.ptr_ = nullptr;
    return *this;
  }
  CLxObjectSPtr& operator = (nullptr_t p) {
    InternalRelease(ptr_);
    ptr_ = nullptr;
    return *this;
  }
  CLxObjectSPtr operator <<= (ObjType p) {
    Attach(p);
    return *this;
  }
  CLxObjectSPtr operator <<= (CLxObjectSPtr &p) {
    Attach(p.ptr_);
    p.ptr_ = nullptr;
    return *this;
  }
  bool operator==(ObjType p) const { return ptr_ == p; }
  bool operator==(const CLxObjectSPtr &p) const { return ptr_ == p.ptr_; }
  bool operator==(nullptr_t) const { return ptr_ == nullptr; }
  bool operator!=(ObjType p) const { return ptr_ != p; }
  bool operator!=(const CLxObjectSPtr &p) const { return ptr_ != p.ptr_; }
  bool operator!=(nullptr_t) const { return ptr_ != nullptr; }

  operator ObjType() const { return ptr_; }
  ObjType* operator&() {
    return &ptr_;
  }
  ObjType const* operator &() const { return &ptr_; }

  ObjType* GetAddressOf() {
    return &ptr_;
  }
  ObjType const *GetAddressOf() const {
    return &ptr_;
  }
  ObjType *ReleaseAndGetAddressOf() {
    InternalRelease(ptr_);
    ptr_ = nullptr;
    return &ptr_;
  }
protected:
  CLX_REFRET_T InternalAddRef(ObjType p) {
    CLX_REFRET_T ref = CL_SUCCESS;
    if(p)
      ref = g_CLxObjectRefcountMgrTable[(int)TypeIndex].AddRef((void *)p);
    return ref;
  }
  CLX_REFRET_T InternalRelease(ObjType p) {
    CLX_REFRET_T ref = CL_SUCCESS;
    if(p)
      ref = g_CLxObjectRefcountMgrTable[(int)TypeIndex].Release((void *)p);
    return ref;
  }

  ObjType ptr_;
};

using ycl_platform_id = CLxObjectSPtr<cl_platform_id, CLX_OBJECT_TYPE::PLATFORM_ID>;
using ycl_device_id = CLxObjectSPtr<cl_device_id, CLX_OBJECT_TYPE::DEVICE_ID>;
using ycl_context = CLxObjectSPtr<cl_context, CLX_OBJECT_TYPE::CONTEXT>;
using ycl_program = CLxObjectSPtr<cl_program, CLX_OBJECT_TYPE::PROGRAM>;
using ycl_kernel = CLxObjectSPtr<cl_kernel, CLX_OBJECT_TYPE::KERNEL>;
using ycl_command_queue = CLxObjectSPtr<cl_command_queue, CLX_OBJECT_TYPE::COMMAND_QUEUE>;
using ycl_event = CLxObjectSPtr<cl_event, CLX_OBJECT_TYPE::EVENT>;
using ycl_mem = CLxObjectSPtr<cl_mem, CLX_OBJECT_TYPE::MEM_OBJECT>;
using ycl_buffer = CLxObjectSPtr<cl_mem, CLX_OBJECT_TYPE::BUFFER>;
using ycl_image = CLxObjectSPtr<cl_mem, CLX_OBJECT_TYPE::IMAGE>;
using ycl_sampler = CLxObjectSPtr<cl_sampler, CLX_OBJECT_TYPE::SAMPLER>;

extern CLHRESULT CheckCLExtensions(cl_device_id device, std::initializer_list<const char *> req_extensions);

/**
 * @param preferred_plats is a string of string terminated with nullptr.
 * 
 */
extern CLHRESULT FindOpenCLPlatform(cl_device_type dev_type,
                                    std::initializer_list<const char *> preferred_plats,
                                    std::initializer_list<const char *> req_extensions,
                                    cl_platform_id *plat_id,
                                    cl_device_id *device);

extern CLHRESULT FindOpenCLPlatform2(cl_device_type dev_type,
                                     std::initializer_list<const char *> preferred_plats,
                                     std::initializer_list<const char *> req_extensions,
                                     void *glrc,  // Optional
                                     void *glsurface, // Optional
                                     cl_platform_id *plat_id,
                                     cl_device_id *device);

extern CLHRESULT CreateDeviceContext(
  cl_platform_id plat_id, cl_device_id dev, cl_context *dev_ctx);

extern CLHRESULT CreateCommandQueue(cl_context dev_ctx, cl_device_id device, cl_command_queue *cmd_queue);

extern CLHRESULT CreateProgramFromSource(
  cl_context context, cl_device_id device, const char *defines, const char *source, size_t src_len, cl_program *program);

extern CLHRESULT CreateProgramFromFile(
    cl_context context, cl_device_id device, const char *defines, const char *fname, cl_program *program);

template<typename ...TArgs> struct __SetKernelArgumentsImpl {
private:
  template <class T>
  CLHRESULT _SetKernelArg(cl_kernel kernel, size_t i, const T &arg) const {
    size_t arg_len = sizeof(*arg);
    return clSetKernelArg(kernel, i, arg_len, arg);
  }
  template <>
  CLHRESULT _SetKernelArg<size_t>(cl_kernel kernel, size_t i, const size_t &arg) const {
    return clSetKernelArg(kernel, i, arg, nullptr);
  }
public:
  template<size_t ...I>
  CLHRESULT Set(cl_kernel kernel, std::index_sequence<I...> indices, const TArgs& ...args) {
    CLHRESULT hr;
    hr = ((_SetKernelArg(kernel, I, args)) | ...);
    return hr;
  }
};

template<typename ...TArgs>
CLHRESULT SetKernelArguments(cl_kernel kernel, const TArgs& ...args) {

  __SetKernelArgumentsImpl<TArgs...> __impl;

  return __impl.Set(kernel, std::make_index_sequence<sizeof...(TArgs)>{}, args...);
}
