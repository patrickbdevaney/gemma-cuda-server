// safetensors.h — minimal mmap reader for safetensors files (header-only).
// Format: [u64 LE header_len][JSON header][raw tensor bytes]. Offsets are relative to data start.
// Supports the dtypes present in the Gemma-4 NVFP4 checkpoint: U8 (packed FP4), F8_E4M3, BF16, F16, F32, I32, I64.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

namespace st {

struct Tensor {
    std::string dtype;                 // "U8","F8_E4M3","BF16","F16","F32","I32",...
    std::vector<int64_t> shape;
    const uint8_t* data = nullptr;     // pointer into mmap
    size_t nbytes = 0;
    int64_t numel() const { int64_t n = 1; for (auto s : shape) n *= s; return n; }
};

// --- tiny JSON parser, just enough for safetensors headers ---
struct JsonP {
    const char* p; const char* end;
    JsonP(const char* s, size_t n) : p(s), end(s + n) {}
    void ws() { while (p < end && (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r')) ++p; }
    bool eat(char c) { ws(); if (p < end && *p == c) { ++p; return true; } return false; }
    std::string str() {
        ws(); if (*p != '"') throw std::runtime_error("json: expected string");
        ++p; std::string s;
        while (p < end && *p != '"') { if (*p == '\\') { ++p; } s.push_back(*p); ++p; }
        ++p; return s;
    }
    double num() {
        ws(); char* e; double v = strtod(p, &e); p = e; return v;
    }
    void skip_value();  // forward
};

inline void JsonP::skip_value() {
    ws();
    if (*p == '"') { str(); }
    else if (*p == '{') { ++p; ws(); if (eat('}')) return; do { str(); eat(':'); skip_value(); } while (eat(',')); eat('}'); }
    else if (*p == '[') { ++p; ws(); if (eat(']')) return; do { skip_value(); } while (eat(',')); eat(']'); }
    else { while (p < end && *p != ',' && *p != '}' && *p != ']') ++p; }
}

class SafeTensors {
public:
    explicit SafeTensors(const std::string& path) {
        fd_ = open(path.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("open failed: " + path);
        struct stat sb; fstat(fd_, &sb); filesize_ = sb.st_size;
        base_ = (const uint8_t*)mmap(nullptr, filesize_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (base_ == MAP_FAILED) throw std::runtime_error("mmap failed: " + path);
        uint64_t hlen; memcpy(&hlen, base_, 8);
        const char* hdr = (const char*)base_ + 8;
        const uint8_t* data_start = base_ + 8 + hlen;
        data_start_ = data_start; data_bytes_ = filesize_ - (8 + hlen);
        parse_header(hdr, hlen, data_start);
    }
    ~SafeTensors() { if (base_ && base_ != MAP_FAILED) munmap((void*)base_, filesize_); if (fd_ >= 0) close(fd_); }

    const uint8_t* dataStart() const { return data_start_; }
    size_t dataBytes() const { return data_bytes_; }
    bool has(const std::string& name) const { return tensors_.count(name) > 0; }
    const Tensor& get(const std::string& name) const {
        auto it = tensors_.find(name);
        if (it == tensors_.end()) throw std::runtime_error("tensor not found: " + name);
        return it->second;
    }
    const std::unordered_map<std::string, Tensor>& all() const { return tensors_; }
    size_t count() const { return tensors_.size(); }

private:
    void parse_header(const char* hdr, size_t hlen, const uint8_t* data_start) {
        JsonP j(hdr, hlen);
        if (!j.eat('{')) throw std::runtime_error("header: expected object");
        j.ws(); if (j.eat('}')) return;
        do {
            std::string name = j.str(); j.eat(':');
            if (name == "__metadata__") { j.skip_value(); continue; }
            j.eat('{');
            Tensor t; int64_t off0 = 0, off1 = 0;
            do {
                std::string key = j.str(); j.eat(':');
                if (key == "dtype") t.dtype = j.str();
                else if (key == "shape") {
                    j.eat('['); j.ws();
                    if (!j.eat(']')) { do { t.shape.push_back((int64_t)j.num()); } while (j.eat(',')); j.eat(']'); }
                } else if (key == "data_offsets") {
                    j.eat('['); off0 = (int64_t)j.num(); j.eat(','); off1 = (int64_t)j.num(); j.eat(']');
                } else j.skip_value();
            } while (j.eat(','));
            j.eat('}');
            t.data = data_start + off0; t.nbytes = (size_t)(off1 - off0);
            tensors_.emplace(std::move(name), std::move(t));
        } while (j.eat(','));
    }
    int fd_ = -1; size_t filesize_ = 0; const uint8_t* base_ = nullptr;
    const uint8_t* data_start_ = nullptr; size_t data_bytes_ = 0;
    std::unordered_map<std::string, Tensor> tensors_;
};

} // namespace st
