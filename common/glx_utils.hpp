#ifndef __COMMON_UTILS_HPP__
#define __COMMON_UTILS_HPP__

#include "assert.h"
#include <cstddef> // For offsetof
#include <type_traits> // For std::enable_if
#include <initializer_list> // For std::initalizer_list
#include <glad/glad.h>

#define GTX_ASSERT(op) assert(op)

namespace GLx {

enum class GLHandleTypeIndex {
  VAO,
  BUFFER,
  TEXTURE,
  FRMAE_BUFFER,
  RENDER_BUFFER,
  HANDLE_TYPE_COUNT,
};

struct GLHandleCreateFreeObjectsFuncsEntry {
  void(__stdcall *const gen)(GLsizei, GLuint *);
  void(__stdcall *const create)(GLsizei, GLuint *);
  void(__stdcall *const free)(GLsizei, const GLuint *);
};
extern GLHandleCreateFreeObjectsFuncsEntry const (
    &GLHandleCreateFreeObjectsFuncsTable())[(unsigned int)GLHandleTypeIndex::HANDLE_TYPE_COUNT];

template <GLHandleTypeIndex Index> struct GLHandle;

template <GLHandleTypeIndex Index> struct GLHandleRef {
public:
  GLHandleRef(GLuint h = 0) : handle(h) {}
  GLHandleRef(const GLHandle<Index> &src) : handle(src.handle) {}
  GLHandleRef &operator=(const GLHandle<Index> &src) {
    handle = src.handle;
    return *this;
  }
  GLHandleRef &operator=(nullptr_t) {
    handle = 0;
    return *this;
  }
  operator bool() const { return handle != 0; }
  operator GLuint() const { return handle; }
  bool operator!=(nullptr_t) const { return handle != 0; }
  bool operator==(nullptr_t) const { return handle == 0; }
  bool operator!=(GLuint h) const { return handle != h; }
  bool operator==(GLuint h) const { return handle == h; }
  bool operator<(const GLHandleRef &rhs) const { return handle < rhs.handle; }
  bool operator>(const GLHandleRef &rhs) const { return handle > rhs.handle; }

private:
  GLuint handle;
};

template <GLHandleTypeIndex Index> struct GLHandle {
public:
  friend struct GLHandleRef<Index>;
  static GLHandle New() {
    GLuint h;
    GLHandleCreateFreeObjectsFuncsTable()[(unsigned int)Index].gen(1, &h);
    return GLHandle(h);
  }

  template<typename = std::enable_if<Index != GLHandleTypeIndex::TEXTURE>>
  static GLHandle Create() {
    GLuint h;
    GLHandleCreateFreeObjectsFuncsTable()[(unsigned int)Index].create(1, &h);
    return GLHandle(h);
  }

  template<typename = std::enable_if<Index == GLHandleTypeIndex::TEXTURE>>
  static GLHandle Create(GLenum type) {
    GLuint h;
    ((void(__stdcall *)(GLenum, GLsizei, GLuint *))GLHandleCreateFreeObjectsFuncsTable()[(unsigned int)Index].create)
      (type, 1, &h);
    return GLHandle(h);
  }

  GLHandle(GLuint h = 0) : handle(h) {}
  GLHandle(const GLHandle &) = delete;
  GLHandle(GLHandle &&rhs) {
    handle = rhs.handle;
    rhs.handle = 0;
  }
  ~GLHandle() { DestroyObject(); }

  void operator=(const GLHandle &) = delete;
  GLHandle &operator=(GLHandle &&rhs) {
    DestroyObject();
    handle = rhs.handle;
    rhs.handle = 0;
    return *this;
  }
  GLHandle &operator=(nullptr_t) {
    DestroyObject();
    return *this;
  }
  operator bool() const { return handle != 0; }
  operator GLuint() const { return handle; }
  operator GLHandleRef<Index>() const {
    return GLHandleRef<Index>(handle);
  }
  bool operator!=(nullptr_t) const { return handle != 0; }
  bool operator==(nullptr_t) const { return handle == 0; }
  bool operator!=(GLuint h) const { return handle != h; }
  bool operator==(GLuint h) const { return handle == h; }
  bool operator<(const GLHandle &rhs) const { return handle < rhs.handle; }
  bool operator>(const GLHandle &rhs) const { return handle > rhs.handle; }

private:
  void DestroyObject() {
    if (handle) {
      GLHandleCreateFreeObjectsFuncsTable()[(unsigned int)Index].free(1, &handle);
      handle = 0;
    }
  }
  GLuint handle;
};

using GLVAO = GLHandle<GLHandleTypeIndex::VAO>;
using GLVAORef = GLHandleRef<GLHandleTypeIndex::VAO>;

using GLBuffer = GLHandle<GLHandleTypeIndex::BUFFER>;
using GLBufferRef = GLHandleRef<GLHandleTypeIndex::BUFFER>;

using GLTexture = GLHandle<GLHandleTypeIndex::TEXTURE>;
using GLTextureRef = GLHandleRef<GLHandleTypeIndex::TEXTURE>;

using GLFrameBuffer = GLHandle<GLHandleTypeIndex::FRMAE_BUFFER>;
using GLFrameBufferRef = GLHandleRef<GLHandleTypeIndex::FRMAE_BUFFER>;

using GLRenderBuffer = GLHandle<GLHandleTypeIndex::RENDER_BUFFER>;
using GLRenderBufferRef = GLHandleRef<GLHandleTypeIndex::RENDER_BUFFER>;

int CheckGLExtensions(std::initializer_list<const char *> reqExtensions);

}; // namespace GLx

#endif // __COMMON_UTILS_HPP__