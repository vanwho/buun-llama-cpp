#include "llama-kv-cache.h"

#undef NDEBUG
#include <cassert>
#include <cstdint>

static void test_dense_source_is_logical_prefix() {
    // This is the failure that caused dense Turbo4 MTP verification to read
    // unwritten rows from the full cache allocation.
    assert(llama_kv_cache_context::attention_source_rows(false, 256, 4096) == 256);
    assert(llama_kv_cache_context::attention_source_rows(false, 4096, 256) == 4096);
}

static void test_paged_source_keeps_physical_window() {
    // Selected attention still needs the physical window from which its
    // validated resident-page view is cropped.
    assert(llama_kv_cache_context::attention_source_rows(true, 256, 4096) == 4096);
    assert(llama_kv_cache_context::attention_source_rows(true, 4096, 256) == 4096);
}

int main() {
    test_dense_source_is_logical_prefix();
    test_paged_source_keeps_physical_window();
    return 0;
}
