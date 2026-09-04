#include "server-context.h"

#include <cstdio>

static bool pager_metrics_export_regression() {
    server_task_result_metrics result;
    result.metrics.pager_metrics = json{
        {"mode", "selective"},
        {"page_capacity", 7},
    };

    const std::string rendered = result.to_metrics();
    return rendered.find("llamacpp:kv_pager_mode{mode=\"selective\"} 1") != std::string::npos &&
           rendered.find("llamacpp:kv_pager_page_capacity 7") != std::string::npos;
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
        pager_metrics_export_regression();
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
