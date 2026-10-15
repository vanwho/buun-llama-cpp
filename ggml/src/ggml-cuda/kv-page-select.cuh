#pragma once

#include "common.cuh"

void ggml_cuda_op_kv_page_select(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_kv_page_rank(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_kv_page_rerank(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_kv_page_mass(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_kv_query_accumulate(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_kv_query_probes(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
