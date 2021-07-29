#include "glx_utils.hpp"
#include <algorithm>
#include <stdio.h>

namespace GLx {

GLHandleCreateFreeObjectsFuncsEntry const (
    &GLHandleCreateFreeObjectsFuncsTable())[(unsigned int)GLHandleTypeIndex::HANDLE_TYPE_COUNT] {
  static GLHandleCreateFreeObjectsFuncsEntry const
      _Entries[(unsigned int)GLHandleTypeIndex::HANDLE_TYPE_COUNT] = {
          {glGenVertexArrays, glCreateVertexArrays, glDeleteVertexArrays},
          {glGenBuffers, glCreateBuffers, glDeleteBuffers},
          {glGenTextures, (void(__stdcall *)(GLsizei, GLuint *))glCreateTextures, glDeleteTextures},
          {glGenFramebuffers, glCreateFramebuffers, glDeleteFramebuffers},
          {glGenRenderbuffers, glCreateRenderbuffers, glDeleteRenderbuffers},
      };

  return _Entries;
}

int CheckGLExtensions(std::initializer_list<const char *> reqExtensions) {

  int reqCount = (int)reqExtensions.size();
  GLint n;
  const char *extension;
  static_assert(sizeof(char) == sizeof(GLubyte), "GL byte is not compatible with char!");

  glGetIntegerv(GL_NUM_EXTENSIONS, &n);

  printf("Enumerate OpenGL extensions:\n");

  for (GLint i = 0; i < n; ++i) {
    extension = (const char *)glGetStringi(GL_EXTENSIONS, i);
    printf("  %s\n", extension);
    if (std::find_if(std::begin(reqExtensions), std::end(reqExtensions), [extension](auto val) {
          return _stricmp(extension, val) == 0;
        }) != std::end(reqExtensions)) {
      reqCount -= 1;
    }
    if (reqCount <= 0)
      break;
  }
  printf("Enumerate OpenGL extensions Done.\n\n");

  return reqCount > 0 ? -1 : 0;
}

} // namespace GLx