// Minimal JSON reader for the rt_* tracks (reflection address_sources, shared BVH vectors). Numbers
// keep their integer value exactly up to 2^53; no unicode escapes beyond pass-through.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

namespace playgta6::gpu::rt_json {

struct Value {
    enum class Type { Null, Bool, Number, String, Array, Object } type = Type::Null;
    bool b = false;
    double num = 0;
    std::string str;
    std::vector<Value> arr;
    std::map<std::string, Value> obj;

    const Value* get(const std::string& k) const {
        const auto it = obj.find(k);
        return it == obj.end() ? nullptr : &it->second;
    }
    uint64_t u64(uint64_t def = 0) const {
        return type == Type::Number && num >= 0 ? static_cast<uint64_t>(num) : def;
    }
};

class Parser {
public:
    explicit Parser(const std::string& s) : s_(s) {}
    bool parse(Value* out, std::string* err) {
        ws();
        if (!value(out, 0)) {
            if (err) *err = "parse error at byte " + std::to_string(p_);
            return false;
        }
        ws();
        if (p_ != s_.size()) {
            if (err) *err = "trailing characters";
            return false;
        }
        return true;
    }

private:
    const std::string& s_;
    size_t p_ = 0;

    void ws() {
        while (p_ < s_.size() && (s_[p_] == ' ' || s_[p_] == '\t' || s_[p_] == '\n' || s_[p_] == '\r')) ++p_;
    }
    bool lit(const char* w) {
        size_t n = 0;
        while (w[n]) ++n;
        if (s_.compare(p_, n, w) != 0) return false;
        p_ += n;
        return true;
    }
    bool str(std::string* out) {
        if (p_ >= s_.size() || s_[p_] != '"') return false;
        ++p_;
        out->clear();
        while (p_ < s_.size() && s_[p_] != '"') {
            char c = s_[p_++];
            if (c == '\\' && p_ < s_.size()) {
                const char e = s_[p_++];
                c = e == 'n' ? '\n' : e == 't' ? '\t' : e == 'r' ? '\r' : e;
            }
            out->push_back(c);
        }
        if (p_ >= s_.size()) return false;
        ++p_;
        return true;
    }
    bool close(char ch) {
        if (p_ < s_.size() && s_[p_] == ch) {
            ++p_;
            return true;
        }
        return false;
    }
    bool value(Value* v, int depth) {
        if (depth > 32 || p_ >= s_.size()) return false;
        const char c = s_[p_];
        if (c == '{') {
            ++p_;
            v->type = Value::Type::Object;
            ws();
            if (close('}')) return true;
            for (;;) {
                ws();
                std::string k;
                if (!str(&k)) return false;
                ws();
                if (!close(':')) return false;
                ws();
                if (!value(&v->obj[k], depth + 1)) return false;
                ws();
                if (close(',')) continue;
                return close('}');
            }
        }
        if (c == '[') {
            ++p_;
            v->type = Value::Type::Array;
            ws();
            if (close(']')) return true;
            for (;;) {
                ws();
                v->arr.emplace_back();
                if (!value(&v->arr.back(), depth + 1)) return false;
                ws();
                if (close(',')) continue;
                return close(']');
            }
        }
        if (c == '"') {
            v->type = Value::Type::String;
            return str(&v->str);
        }
        if (lit("true")) {
            v->type = Value::Type::Bool;
            v->b = true;
            return true;
        }
        if (lit("false")) {
            v->type = Value::Type::Bool;
            return true;
        }
        if (lit("null")) return true;
        char* end = nullptr;
        v->num = std::strtod(s_.c_str() + p_, &end);
        if (end == s_.c_str() + p_) return false;
        p_ = static_cast<size_t>(end - s_.c_str());
        v->type = Value::Type::Number;
        return true;
    }
};

inline bool parse(const std::string& s, Value* out, std::string* err = nullptr) {
    return Parser(s).parse(out, err);
}

}  // namespace playgta6::gpu::rt_json
