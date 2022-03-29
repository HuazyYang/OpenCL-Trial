#pragma once
#include <exception>

namespace irrschitz {

class VkException : std::exception {
VkException(const char *msg): exception(msg), _code(-1) {}
VkException(const char *msg, int code): exception(msg), _code(code) {}

protected:
  int _code;
};



};