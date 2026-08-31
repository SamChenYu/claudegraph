// json.hpp — a small, dependency-free JSON parser, just enough to read the
// fields we need out of Claude Code's JSONL transcripts and history file.
#ifndef CONVOGRAPH_JSON_HPP
#define CONVOGRAPH_JSON_HPP

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace mjson {

struct Value {
  enum Type { Null, Bool, Num, Str, Arr, Obj } type = Null;
  bool b = false;
  double num = 0;
  std::string str;
  std::vector<Value> arr;
  std::map<std::string, Value> obj;

  const Value* find(const std::string& key) const {
    if (type != Obj)
      return nullptr;
    auto it = obj.find(key);
    return it == obj.end() ? nullptr : &it->second;
  }
  std::string as_str(const std::string& def = "") const {
    return type == Str ? str : def;
  }
};

class Parser {
 public:
  explicit Parser(const std::string& s) : s_(s) {}
  bool parse(Value& out) {
    ws();
    bool ok = value(out);
    return ok;
  }

 private:
  const std::string& s_;
  size_t i_ = 0;

  void ws() {
    while (i_ < s_.size()) {
      char c = s_[i_];
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r')
        i_++;
      else
        break;
    }
  }
  bool value(Value& v) {
    ws();
    if (i_ >= s_.size())
      return false;
    char c = s_[i_];
    switch (c) {
      case '{':
        return object(v);
      case '[':
        return array(v);
      case '"':
        v.type = Value::Str;
        return string(v.str);
      case 't':
      case 'f':
        return boolean(v);
      case 'n':
        return null(v);
      default:
        return number(v);
    }
  }
  bool object(Value& v) {
    v.type = Value::Obj;
    i_++;  // {
    ws();
    if (i_ < s_.size() && s_[i_] == '}') {
      i_++;
      return true;
    }
    while (true) {
      ws();
      if (i_ >= s_.size() || s_[i_] != '"')
        return false;
      std::string key;
      if (!string(key))
        return false;
      ws();
      if (i_ >= s_.size() || s_[i_] != ':')
        return false;
      i_++;
      Value child;
      if (!value(child))
        return false;
      v.obj[key] = std::move(child);
      ws();
      if (i_ >= s_.size())
        return false;
      if (s_[i_] == ',') {
        i_++;
        continue;
      }
      if (s_[i_] == '}') {
        i_++;
        return true;
      }
      return false;
    }
  }
  bool array(Value& v) {
    v.type = Value::Arr;
    i_++;  // [
    ws();
    if (i_ < s_.size() && s_[i_] == ']') {
      i_++;
      return true;
    }
    while (true) {
      Value child;
      if (!value(child))
        return false;
      v.arr.push_back(std::move(child));
      ws();
      if (i_ >= s_.size())
        return false;
      if (s_[i_] == ',') {
        i_++;
        continue;
      }
      if (s_[i_] == ']') {
        i_++;
        return true;
      }
      return false;
    }
  }
  static void utf8(uint32_t cp, std::string& out) {
    if (cp <= 0x7F) {
      out += char(cp);
    } else if (cp <= 0x7FF) {
      out += char(0xC0 | (cp >> 6));
      out += char(0x80 | (cp & 0x3F));
    } else if (cp <= 0xFFFF) {
      out += char(0xE0 | (cp >> 12));
      out += char(0x80 | ((cp >> 6) & 0x3F));
      out += char(0x80 | (cp & 0x3F));
    } else {
      out += char(0xF0 | (cp >> 18));
      out += char(0x80 | ((cp >> 12) & 0x3F));
      out += char(0x80 | ((cp >> 6) & 0x3F));
      out += char(0x80 | (cp & 0x3F));
    }
  }
  int hex4() {
    if (i_ + 4 > s_.size())
      return -1;
    int v = 0;
    for (int k = 0; k < 4; k++) {
      char c = s_[i_++];
      v <<= 4;
      if (c >= '0' && c <= '9')
        v |= c - '0';
      else if (c >= 'a' && c <= 'f')
        v |= c - 'a' + 10;
      else if (c >= 'A' && c <= 'F')
        v |= c - 'A' + 10;
      else
        return -1;
    }
    return v;
  }
  bool string(std::string& out) {
    i_++;  // opening quote
    while (i_ < s_.size()) {
      char c = s_[i_++];
      if (c == '"')
        return true;
      if (c == '\\') {
        if (i_ >= s_.size())
          return false;
        char e = s_[i_++];
        switch (e) {
          case '"': out += '"'; break;
          case '\\': out += '\\'; break;
          case '/': out += '/'; break;
          case 'b': out += '\b'; break;
          case 'f': out += '\f'; break;
          case 'n': out += '\n'; break;
          case 'r': out += '\r'; break;
          case 't': out += '\t'; break;
          case 'u': {
            int cp = hex4();
            if (cp < 0)
              return false;
            if (cp >= 0xD800 && cp <= 0xDBFF) {  // high surrogate
              if (i_ + 1 < s_.size() && s_[i_] == '\\' && s_[i_ + 1] == 'u') {
                i_ += 2;
                int lo = hex4();
                if (lo >= 0xDC00 && lo <= 0xDFFF)
                  cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
              }
            }
            utf8((uint32_t)cp, out);
            break;
          }
          default:
            out += e;
        }
      } else {
        out += c;
      }
    }
    return false;
  }
  bool boolean(Value& v) {
    if (s_.compare(i_, 4, "true") == 0) {
      v.type = Value::Bool;
      v.b = true;
      i_ += 4;
      return true;
    }
    if (s_.compare(i_, 5, "false") == 0) {
      v.type = Value::Bool;
      v.b = false;
      i_ += 5;
      return true;
    }
    return false;
  }
  bool null(Value& v) {
    if (s_.compare(i_, 4, "null") == 0) {
      v.type = Value::Null;
      i_ += 4;
      return true;
    }
    return false;
  }
  bool number(Value& v) {
    size_t start = i_;
    if (i_ < s_.size() && (s_[i_] == '-' || s_[i_] == '+'))
      i_++;
    bool any = false;
    while (i_ < s_.size() &&
           ((s_[i_] >= '0' && s_[i_] <= '9') || s_[i_] == '.' || s_[i_] == 'e' ||
            s_[i_] == 'E' || s_[i_] == '+' || s_[i_] == '-')) {
      i_++;
      any = true;
    }
    if (!any)
      return false;
    v.type = Value::Num;
    v.num = std::strtod(s_.c_str() + start, nullptr);
    return true;
  }
};

inline bool parse_line(const std::string& line, Value& out) {
  Parser p(line);
  return p.parse(out);
}

}  // namespace mjson

#endif  // CONVOGRAPH_JSON_HPP
