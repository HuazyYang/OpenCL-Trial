// File:int128IO.cpp
#include "int128.h"
#include <ctype.h>
#include <string.h>
#include <assert.h>
#include <exception>

inline char radixLetter(unsigned int c) {
  return (c < 10) ? ('0' + c) : ('a' + (c-10));
}

inline wchar_t wradixLetter(unsigned int c) {
  return (c < 10) ? ('0' + c) : ('a' + (c-10));
}

inline bool isodigit(unsigned char ch) {
  return ('0' <= ch) && (ch < '8');
}

unsigned int convertNumberChar(char digit);

static const _int128 _s0(0);
static const _int128 _s10(10);
static const _int128 _s16(16);
static const _int128 _s8(16);

char *_i128toa(_int128 value, char *str, int radix) {
  assert(radix >= 2 && radix <= 36);
  char *s = str;
  const bool negative = value < _s0;
  if (negative && (radix == 10)) {
    value = -value;
    while (value != _s0) {
      const unsigned int c = value % _s10;
      *(s++) = radixLetter(c);
      value /= _s10;
    }
    *(s++) = '-';
    *s = 0;
    return _strrev(str);
  }

  _uint128 v(value);
  const _uint128 r(radix);
  while (v != _s0) {
    const unsigned int c = v % r;
    *(s++) = radixLetter(c);
    v /= r;
  }
  if (s == str) {
    return strcpy(str, "0");
  }
  else {
    *s = 0;
    return _strrev(str);
  }
  return str;
}

wchar_t *_i128tow(_int128 value, wchar_t *str, int radix) {
  wchar_t *s = str;
  const bool negative = value < _s0;
  if (negative && (radix == 10)) {
    value = -value;
    while (value != _s0) {
      const unsigned int c = value % _s10;
      *(s++) = wradixLetter(c);
      value /= _s10;
    }
    *(s++) = '-';
    *s = 0;
    return _wcsrev(str);
  }

  _uint128 v(value);
  const _uint128 r(radix);
  while (v != _s0) {
    const unsigned int c = v % r;
    *(s++) = radixLetter(c);
    v /= r;
  }
  if (s == str) {
    return wcscpy(str, L"0");
  }
  else {
    *s = 0;
    return _wcsrev(str);
  }
  return str;
}

const char *_int128::parseDec(const char *str) { // return pointer to char following the number
  bool negative = false;
  bool gotDigit = false;
  switch (*str) {
  case '+':
    str++;
    break;
  case '-':
    str++;
    negative = true;
  }
  *this = _s0;
  while (isdigit(*str)) {
    gotDigit = true;
    const unsigned int d = *(str++) - '0';
    *this *= _s10;
    *this += d;
  }
  if (!gotDigit) {
    throw "_int128:string is not a number";
  }
  if (negative) {
    *this = -*this;
  }
  return str;
}

const char *_int128::parseHex(const char *str) {
  *this = 0;
  while (isxdigit(*str)) {
    const unsigned int d = convertNumberChar(*(str++));
    *this *= _s16;
    *this += d;
  }
  return str;
}

const char *_int128::parseOct(const char *str) {
  *this = 0;
  while (isodigit(*str)) {
    const unsigned int d = convertNumberChar(*(str++));
    *this *= _s8;
    *this += d;
  }
  return str;
}

_int128::_int128(const char *str) {
  if (*str == '-') {
    parseDec(str);
  } else {
    if (!isdigit(*str)) {
      throw std::exception("_int128:string is not an integer");
    }
    if (*str == '0') {
      switch (str[1]) {
      case 'x':
        parseHex(str + 2);
        break;
      case 0:
        *this = 0;
        break;
      default:
        parseOct(str + 1);
      }
    }
    else {
      parseDec(str);
    }
  }
}

static const _uint128 _0(0);
static const _uint128 _10(10);
static const _uint128 _16(16);
static const _uint128 _8(16);

char*_ui128toa(_uint128 value, char *str, int radix) {
  assert(radix >= 2 && radix <= 36);
  char *s = str;
  const _uint128 r(radix);
  while (value != _0) {
    const unsigned int c = value % r;
    *(s++) = radixLetter(c);
    value /= r;
  }
  if (s == str) {
    return strcpy(str, "0");
  }
  else {
    *s = 0;
    return _strrev(str);
  }
}

wchar_t *_ui128tow(_uint128 value, wchar_t *str, int radix) {
  assert(radix >= 2 && radix <= 36);
  wchar_t *s = str;
  const _uint128 r(radix);
  while (value != _0) {
    const unsigned int c = value % r;
    *(s++) = wradixLetter(c);
    value /= r;
  }
  if (s == str) {
    return wcscpy(str, L"0");
  }
  else {
    *s = 0;
    return _wcsrev(str);
  }
}

const char *_uint128::parseDec(const char *str) {
  *this = 0;
  while (isdigit(*str)) {
    const unsigned int d = *(str++) - '0';
    *this *= _10;
    *this += d;
  }
  return str;
}

const char *_uint128::parseHex(const char *str) {
  *this = 0;
  while (isxdigit(*str)) {
    const unsigned int d = convertNumberChar(*(str++));
    *this *= _16;
    *this += d;
  }
  return str;
}

const char *_uint128::parseOct(const char *str) {
  *this = 0;
  while (isodigit(*str)) {
    const unsigned int d = convertNumberChar(*(str++));
    *this *= _8;
    *this += d;
  }
  return str;
}

_uint128::_uint128(const char *str) {
  if (!isdigit(*str)) {
    throw std::exception("_uint128:string is not an integer");
  }
  if (*str == '0') {
    switch (str[1]) {
    case 'x':
      parseHex(str + 2);
      break;
    case 0:
      *this = 0;
      break;
    default:
      parseOct(str + 1);
      break;
    }
  }
  else {
    parseDec(str);
  }
}

unsigned int convertNumberChar(char digit) {
  switch(digit) {
  case '0': return 0;
  case '1': return 1;
  case '2': return 2;
  case '3': return 3;
  case '4': return 4;
  case '5': return 5;
  case '6': return 6;
  case '7': return 7;
  case '8': return 8;
  case '9': return 9;
  case 'a':
  case 'A': return 10;
  case 'b':
  case 'B': return 11;
  case 'c':
  case 'C': return 12;
  case 'd':
  case 'D': return 13;
  case 'e':
  case 'E': return 14;
  case 'f':
  case 'F': return 15;
  default :
    return 0;
  }
}
