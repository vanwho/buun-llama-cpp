#pragma once

struct server_prompt_trim_recovery_result {
    bool target_trim_attempted = false;
    bool target_trim_succeeded = false;
    bool draft_trim_attempted = false;
    bool draft_trim_succeeded = false;
    bool recovery_attempted = false;
    bool target_reset_succeeded = false;
    bool draft_reset_succeeded = false;
    bool backup_reset_attempted = false;
    bool backup_reset_succeeded = false;

    bool trim_succeeded() const noexcept {
        return target_trim_succeeded &&
            (!draft_trim_attempted || draft_trim_succeeded);
    }

    bool recovery_succeeded() const noexcept {
        return recovery_attempted && target_reset_succeeded &&
            draft_reset_succeeded &&
            (!backup_reset_attempted || backup_reset_succeeded);
    }
};

// Keep the paired memory operation deterministic and directly testable. A
// rejected partial trim may have changed one child before another refused it,
// so recovery always tries a full reset of both contexts and any live backup
// sequence independently.
template <typename BeforeRecovery, typename TargetTrim, typename DraftTrim,
          typename TargetReset, typename DraftReset, typename BackupReset>
server_prompt_trim_recovery_result server_prompt_trim_recover(
        bool has_draft,
        bool has_backup,
        BeforeRecovery && before_recovery,
        TargetTrim && target_trim,
        DraftTrim && draft_trim,
        TargetReset && target_reset,
        DraftReset && draft_reset,
        BackupReset && backup_reset) {
    server_prompt_trim_recovery_result result;
    result.target_trim_attempted = true;
    result.target_trim_succeeded = target_trim();
    if (result.target_trim_succeeded && has_draft) {
        result.draft_trim_attempted = true;
        result.draft_trim_succeeded = draft_trim();
    }

    if (result.trim_succeeded()) {
        return result;
    }

    result.recovery_attempted = true;
    before_recovery();
    result.target_reset_succeeded = target_reset();
    result.draft_reset_succeeded = !has_draft;
    if (has_draft) {
        result.draft_reset_succeeded = draft_reset();
    }
    result.backup_reset_attempted = has_backup;
    result.backup_reset_succeeded = !has_backup;
    if (has_backup) {
        result.backup_reset_succeeded = backup_reset();
    }
    return result;
}
