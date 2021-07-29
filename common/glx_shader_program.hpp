#ifndef __SHADER_PROGRAM_HPP__
#define __SHADER_PROGRAM_HPP__

namespace GLx {

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

};


#endif /*__SHADER_PROGRAM_HPP__ */