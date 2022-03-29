#include "CommonUtils.hpp"
#include <iostream>
#include <algorithm>

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

  std::cout << "Enumerate GL extensions:" << std::endl;

  for (GLint i = 0; i < n; ++i) {
    extension = (const char *)glGetStringi(GL_EXTENSIONS, i);
    std::cout << extension << std::endl;
    if (std::find_if(std::begin(reqExtensions), std::end(reqExtensions), [extension](auto val) {
          return _stricmp(extension, val) == 0;
        }) != std::end(reqExtensions)) {
      reqCount -= 1;
    }
    if (reqCount <= 0)
      break;
  }
  std::cout << "Enumerate GL extensions: Done" << std::endl;

  return reqCount > 0 ? -1 : 0;
}

void GLRSSetDepthStencilState(const DepthStencilState *dss) {
  GLint state;

  if (glGetIntegerv(GL_DEPTH_TEST, &state), state != dss->DepthEnable)
    dss->DepthEnable ? glEnable(GL_DEPTH_TEST) : glDisable(GL_DEPTH_TEST);

  if (glGetIntegerv(GL_DEPTH_FUNC, &state), state != dss->DepthFunc)
    glDepthFunc(dss->DepthFunc);

  if (glGetIntegerv(GL_DEPTH_WRITEMASK, &state), state != dss->DepthWriteMask)
    glDepthMask(dss->DepthWriteMask);

  if (glGetIntegerv(GL_STENCIL_TEST, &state), state != dss->StencilEnable)
    dss->StencilEnable ? glEnable(GL_STENCIL_TEST) : glDisable(GL_STENCIL_TEST);
  // front face
  if ((glGetIntegerv(GL_STENCIL_VALUE_MASK, &state), state != dss->FrontFace.StencilReadMask) ||
      (glGetIntegerv(GL_STENCIL_REF, &state), state != dss->FrontFace.StencilRef) ||
      (glGetIntegerv(GL_STENCIL_FUNC, &state), state != dss->FrontFace.StencilFunc))
    glStencilFuncSeparate(GL_FRONT, dss->FrontFace.StencilFunc, dss->FrontFace.StencilRef,
                          dss->FrontFace.StencilReadMask);

  if (glGetIntegerv(GL_STENCIL_WRITEMASK, &state), state != dss->FrontFace.StencilWriteMask)
    glStencilMaskSeparate(GL_FRONT, dss->FrontFace.StencilWriteMask);
  if ((glGetIntegerv(GL_STENCIL_FAIL, &state), state != dss->FrontFace.StencilFailOp) ||
      (glGetIntegerv(GL_STENCIL_PASS_DEPTH_FAIL, &state),
       state != dss->FrontFace.StencilDepthFailOp) ||
      (glGetIntegerv(GL_STENCIL_PASS_DEPTH_PASS, &state), state != dss->FrontFace.StencilPassOp))
    glStencilOpSeparate(GL_FRONT, dss->FrontFace.StencilFailOp, dss->FrontFace.StencilDepthFailOp,
                        dss->FrontFace.StencilPassOp);

  if (glGetIntegerv(GL_STENCIL_WRITEMASK, &state), state != dss->FrontFace.StencilWriteMask)
    glStencilMaskSeparate(GL_FRONT, dss->FrontFace.StencilWriteMask);
  // back face
  if ((glGetIntegerv(GL_STENCIL_BACK_VALUE_MASK, &state), state != dss->BackFace.StencilReadMask) ||
      (glGetIntegerv(GL_STENCIL_BACK_REF, &state), state != dss->BackFace.StencilRef) ||
      (glGetIntegerv(GL_STENCIL_BACK_FUNC, &state), state != dss->BackFace.StencilFunc))
    glStencilFuncSeparate(GL_BACK, dss->BackFace.StencilFunc, dss->BackFace.StencilRef,
                          dss->BackFace.StencilReadMask);

  if (glGetIntegerv(GL_STENCIL_BACK_WRITEMASK, &state), state != dss->BackFace.StencilWriteMask)
    glStencilMaskSeparate(GL_BACK, dss->BackFace.StencilWriteMask);
  if ((glGetIntegerv(GL_STENCIL_BACK_FAIL, &state), state != dss->BackFace.StencilFailOp) ||
      (glGetIntegerv(GL_STENCIL_BACK_PASS_DEPTH_FAIL, &state),
       state != dss->BackFace.StencilDepthFailOp) ||
      (glGetIntegerv(GL_STENCIL_BACK_PASS_DEPTH_PASS, &state),
       state != dss->BackFace.StencilPassOp))
    glStencilOpSeparate(GL_BACK, dss->BackFace.StencilFailOp, dss->BackFace.StencilDepthFailOp,
                        dss->BackFace.StencilPassOp);

  if (glGetIntegerv(GL_STENCIL_BACK_WRITEMASK, &state), state != dss->BackFace.StencilWriteMask)
    glStencilMaskSeparate(GL_BACK, dss->BackFace.StencilWriteMask);
}
} // namespace GLx