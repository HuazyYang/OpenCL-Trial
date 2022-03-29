#ifndef __SHADER_PROGRAM_HPP__
#define __SHADER_PROGRAM_HPP__

struct GLSLShaderMacro {
  const char *name;
  const char *value;
};

class ShaderProgram {
public:
  ShaderProgram();
  ~ShaderProgram();

  /**
   * Compile shaders and link them into a pipeline program.
   */
  int Create(const char *vsFilePath, const char *fsFilePath);
  int Create(
    const char *vsPath, // VS
    const char *tcsPath, // TCS
    const char *tesPath, // TES
    const char *gsPath, // GS
    const char *fsPath // FS
    );
  int Destory();
  void Use();

  void SetBool(const char *name, bool value);
  void SetInt(const char *name, int value);
  void SetFloat(const char *name, float value);
  void SetVec2(const char *name, const float *value);
  void SetVec3(const char *name, const float *value);
  void SetVec4(const char *name, const float *value);
  void SetMat2x2(const char *name, const float *value);
  void SetMat3x2(const char *name, const float *value);
  void SetMat3x3(const char *name, const float *value);
  void SetMat4(const char *name, const float *value);

protected:
  ShaderProgram(const ShaderProgram&) = delete;
  void operator = (const ShaderProgram &) = delete;

  unsigned int programId_;
};

class ComputeProgram {
public:
  ComputeProgram();
  ~ComputeProgram();

  int Create(const char *filePath);
  void Destroy();
  void Use();

private:
  unsigned int programId_;
};


#endif /*__SHADER_PROGRAM_HPP__ */