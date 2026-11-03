// Small, persistent, CPU-only GGUF tokenizer used by repository chunk maps.
// Protocol: one JSON request and one JSON response per line on stdio.

#include "llama.h"
#include "ggml-backend.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace {

void log_to_stderr(enum ggml_log_level, const char * text, void *) {
    if (text != nullptr) {
        std::fputs(text, stderr);
    }
}

std::vector<llama_token> encode(const llama_vocab * vocab, const std::string & text,
                                bool parse_special) {
    if (text.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error("input exceeds llama_tokenize int32 byte limit");
    }
    const auto text_size = static_cast<int32_t>(text.size());
    int32_t needed = llama_tokenize(vocab, text.data(), text_size, nullptr, 0,
                                    false, parse_special);
    if (needed == std::numeric_limits<int32_t>::min()) {
        throw std::runtime_error("tokenization result exceeds int32 token limit");
    }
    if (needed < 0) {
        needed = -needed;
    }
    std::vector<llama_token> tokens(static_cast<size_t>(needed));
    const int32_t actual = llama_tokenize(vocab, text.data(), text_size,
                                          tokens.data(), needed, false, parse_special);
    if (actual < 0 || actual != needed) {
        throw std::runtime_error("llama_tokenize returned inconsistent token count");
    }
    return tokens;
}

std::string token_piece(const llama_vocab * vocab, llama_token token) {
    std::vector<char> buffer(64);
    int32_t n = llama_token_to_piece(vocab, token, buffer.data(),
                static_cast<int32_t>(buffer.size()), 0, true);
    if (n < 0) {
        if (n == std::numeric_limits<int32_t>::min()) {
            throw std::runtime_error("token piece exceeds int32 byte limit");
        }
        buffer.resize(static_cast<size_t>(-n));
        n = llama_token_to_piece(vocab, token, buffer.data(),
                                 static_cast<int32_t>(buffer.size()), 0, true);
    }
    if (n < 0 || static_cast<size_t>(n) > buffer.size()) {
        throw std::runtime_error("llama_token_to_piece failed");
    }
    return std::string(buffer.data(), static_cast<size_t>(n));
}

bool valid_utf8(const std::string & value) {
    const auto * p = reinterpret_cast<const unsigned char *>(value.data());
    size_t i = 0;
    while (i < value.size()) {
        const unsigned char c = p[i++];
        if (c <= 0x7f) continue;
        uint32_t codepoint = 0;
        size_t continuation = 0;
        if (c >= 0xc2 && c <= 0xdf) { codepoint = c & 0x1f; continuation = 1; }
        else if (c >= 0xe0 && c <= 0xef) { codepoint = c & 0x0f; continuation = 2; }
        else if (c >= 0xf0 && c <= 0xf4) { codepoint = c & 0x07; continuation = 3; }
        else return false;
        if (i + continuation > value.size()) return false;
        for (size_t j = 0; j < continuation; ++j) {
            const unsigned char d = p[i++];
            if ((d & 0xc0) != 0x80) return false;
            codepoint = (codepoint << 6) | (d & 0x3f);
        }
        if ((continuation == 2 && codepoint < 0x800) ||
            (continuation == 3 && codepoint < 0x10000) ||
            (codepoint >= 0xd800 && codepoint <= 0xdfff) || codepoint > 0x10ffff) {
            return false;
        }
    }
    return true;
}

std::string hex_bytes(const std::string & value) {
    static constexpr char hex[] = "0123456789abcdef";
    std::string out;
    out.reserve(value.size() * 2);
    for (unsigned char c : value) {
        out.push_back(hex[c >> 4]);
        out.push_back(hex[c & 0x0f]);
    }
    return out;
}

std::vector<llama_token> json_tokens(const json & value) {
    if (!value.is_array()) throw std::runtime_error("tokens must be an array");
    std::vector<llama_token> out;
    out.reserve(value.size());
    for (const auto & item : value) {
        if (!item.is_number_integer()) throw std::runtime_error("token IDs must be integers");
        const int64_t token = item.get<int64_t>();
        if (token < std::numeric_limits<llama_token>::min() ||
            token > std::numeric_limits<llama_token>::max()) {
            throw std::runtime_error("token ID is outside llama_token range");
        }
        out.push_back(static_cast<llama_token>(token));
    }
    return out;
}

json handle(const json & request, const llama_vocab * vocab) {
    const auto op_it = request.find("op");
    if (op_it == request.end() || !op_it->is_string()) {
        throw std::runtime_error("request requires string op");
    }
    const std::string op = op_it->get<std::string>();
    if (op == "encode") {
        const auto text_it = request.find("text");
        if (text_it == request.end() || !text_it->is_string()) {
            throw std::runtime_error("encode requires string text");
        }
        const std::string text = text_it->get<std::string>();
        bool parse_special = true;
        const auto parse_it = request.find("parse_special");
        if (parse_it != request.end()) {
            if (!parse_it->is_boolean()) throw std::runtime_error("parse_special must be boolean");
            parse_special = parse_it->get<bool>();
        }
        const auto tokens = encode(vocab, text, parse_special);
        json response = {{"tokens", tokens}};
        const auto pieces_it = request.find("pieces");
        if (pieces_it != request.end() && !pieces_it->is_boolean()) {
            throw std::runtime_error("pieces must be boolean");
        }
        if (pieces_it != request.end() && pieces_it->get<bool>()) {
            json lengths = json::array();
            for (const auto token : tokens) {
                lengths.push_back(token_piece(vocab, token).size());
            }
            response["piece_byte_lengths"] = std::move(lengths);
        }
        return response;
    }
    if (op == "decode") {
        const auto token_it = request.find("tokens");
        if (token_it == request.end()) throw std::runtime_error("decode requires tokens");
        const auto tokens = json_tokens(*token_it);
        std::string text;
        for (const auto token : tokens) text += token_piece(vocab, token);
        if (valid_utf8(text)) return {{"text", text}};
        return {{"bytes_hex", hex_bytes(text)}};
    }
    throw std::runtime_error("op must be encode or decode");
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: repo_tokenizer MODEL.gguf\n");
        return 2;
    }

    llama_log_set(log_to_stderr, nullptr);
    llama_backend_init();

    llama_model_params params = llama_model_default_params();
    params.vocab_only = true;
    params.n_gpu_layers = 0;
    ggml_backend_dev_t cpu_device =
        ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu_device == nullptr) {
        std::fprintf(stderr, "error: CPU ggml device is unavailable\n");
        llama_backend_free();
        return 1;
    }
    ggml_backend_dev_t devices[] = {cpu_device, nullptr};
    params.devices = devices;
    llama_model * model = llama_model_load_from_file(argv[1], params);
    if (model == nullptr) {
        std::fprintf(stderr, "error: cannot load GGUF vocabulary: %s\n", argv[1]);
        llama_backend_free();
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::string line;
    while (std::getline(std::cin, line)) {
        try {
            const json request = json::parse(line);
            std::cout << handle(request, vocab).dump(-1, ' ', false,
                                                     json::error_handler_t::strict) << '\n';
        } catch (const std::exception & error) {
            json response = {{"error", error.what()}};
            std::cout << response.dump() << '\n';
        }
        std::cout.flush();
    }

    llama_model_free(model);
    llama_backend_free();
    return 0;
}
