#include "server-context.h"

#include <cstdio>

static bool pager_metrics_export_regression() {
    server_task_result_metrics result;
    result.metrics.pager_metrics = json{
        {"mode", "selective"},
        {"route", "selected_direct"},
        {"mtp_backend", "gpu"},
        {"target_type_k", "turbo4"},
        {"target_type_v", "turbo4"},
        {"mtp_type_k", "turbo4"},
        {"mtp_type_v", "turbo4"},
        {"page_capacity", 7},
        // Pager snapshots also contain boolean state. Prometheus rendering
        // must ignore it instead of asking nlohmann::json for a double.
        {"admission_accepted", true},
        {"attention_publish_time_us", 11},
        {"attention_d2h_time_us", 13},
        {"waits", 17},
    };

    const std::string rendered = result.to_metrics();
    return rendered.find("llamacpp:kv_pager_mode{mode=\"selective\"} 1") != std::string::npos &&
           rendered.find("llamacpp:kv_pager_route{route=\"selected_direct\"} 1") != std::string::npos &&
           rendered.find("llamacpp:kv_pager_mtp_backend{backend=\"gpu\"} 1") != std::string::npos &&
           rendered.find("llamacpp:kv_pager_target_type_k{type=\"turbo4\"} 1") != std::string::npos &&
           rendered.find("llamacpp:kv_pager_target_type_v{type=\"turbo4\"} 1") != std::string::npos &&
           rendered.find("llamacpp:kv_pager_mtp_type_k{type=\"turbo4\"} 1") != std::string::npos &&
           rendered.find("llamacpp:kv_pager_mtp_type_v{type=\"turbo4\"} 1") != std::string::npos &&
           rendered.find("llamacpp:kv_pager_page_capacity 7") != std::string::npos &&
           rendered.find("llamacpp:kv_pager_admission_accepted") == std::string::npos &&
           rendered.find("llamacpp:kv_pager_attention_publish_time_us 11") != std::string::npos &&
           rendered.find("llamacpp:kv_pager_attention_d2h_time_us 13") != std::string::npos &&
           rendered.find("llamacpp:kv_pager_waits 17") != std::string::npos;
}

static bool pager_metrics_state_regression() {
    server_task_result_metrics result;

    // A disabled server has no pager object at all; this must not turn into a
    // collection of numeric zero gauges.
    result.metrics.pager_metrics = nullptr;
    const std::string disabled = result.to_metrics();
    if (disabled.find("llamacpp:kv_pager_") != std::string::npos) {
        return false;
    }

    // A live MTP context whose allocation cannot satisfy the GPU/Turbo4
    // contract is explicit and distinct from no native MTP context.
    result.metrics.pager_metrics = json{
        {"mode", "selective"},
        {"route", "refusal"},
        {"mtp_backend", "unsupported"},
        {"mtp_type_k", "f16"},
        {"mtp_type_v", "f16"},
    };
    const std::string unsupported = result.to_metrics();
    return unsupported.find("llamacpp:kv_pager_mtp_backend{backend=\"unsupported\"} 1") != std::string::npos &&
           unsupported.find("llamacpp:kv_pager_mtp_type_k{type=\"f16\"} 1") != std::string::npos;
}

int main() {
    const auto result = server_mmproj_lifecycle_for_test();
    const bool passed =
        result.null_binding_clears_views &&
        result.restored_binding_updates_views &&
        result.failed_recreation_stays_null &&
        result.normal_restore_once &&
        result.thrown_media_restore_once &&
        result.thrown_callback_restore_once &&
        result.throwing_restore_not_retried &&
        result.incompatible_draft_disables_shift &&
        result.incompatible_draft_not_shifted &&
        result.compatible_draft_enables_shift &&
        result.compatible_draft_shifted &&
        pager_metrics_export_regression() &&
        pager_metrics_state_regression();
    if (!passed) {
        std::fprintf(stderr,
            "mmproj lifecycle regression failed: "
            "null=%d restored=%d failed=%d normal=%d media=%d callback=%d throwing=%d "
            "draft_off=%d draft_not_shifted=%d draft_on=%d draft_shifted=%d\n",
            result.null_binding_clears_views,
            result.restored_binding_updates_views,
            result.failed_recreation_stays_null,
            result.normal_restore_once,
            result.thrown_media_restore_once,
            result.thrown_callback_restore_once,
            result.throwing_restore_not_retried,
            result.incompatible_draft_disables_shift,
            result.incompatible_draft_not_shifted,
            result.compatible_draft_enables_shift,
            result.compatible_draft_shifted);
        return 1;
    }
    return 0;
}
