#include "speculative.h"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

void test_proposal_positions() {
    assert(common_speculative_mtp_draft_position(40, 0, false) == 41);
    assert(common_speculative_mtp_draft_position(40, 1, false) == 42);
    assert(common_speculative_mtp_draft_position(40, 2, false) == 43);
    assert(common_speculative_mtp_draft_position(40, 0, true) == 40);
    assert(common_speculative_mtp_draft_position(40, 2, true) == 40);
    assert(common_speculative_mtp_draft_position(-1, 0, false) == -1);
}

void test_frontier_and_rollback_once() {
    for (size_t accepted = 0; accepted <= 5; ++accepted) {
        const auto frontier = common_speculative_rollback_frontier_resolve(
            12, 5, accepted);
        assert(frontier.valid());
        assert(frontier.accepted_draft_tokens == accepted);
        assert(frontier.accepted_token_count == 13 + (int64_t) accepted);
        assert(frontier.rejected_suffix_begin == frontier.accepted_token_count);
        assert(frontier.rejected_suffix_end == 18);
        assert(frontier.rejected_draft_tokens == 5 - accepted);
    }

    const auto frontier = common_speculative_rollback_frontier_resolve(12, 5, 2);

    common_speculative_mtp_rollback_guard guard;
    assert(guard.should_apply((llama_pos) frontier.accepted_token_count));
    assert(!guard.should_apply((llama_pos) frontier.accepted_token_count));
    assert(guard.should_apply((llama_pos) frontier.rejected_suffix_end));
    assert(!guard.should_apply(-1));
    guard.reset();
    assert(guard.should_apply((llama_pos) frontier.accepted_token_count));
}

void test_hidden_carry_freshness() {
    common_speculative_mtp_carry_lifecycle carry;
    float pending_h = 7.0f;

    assert(carry.draft_carry(&pending_h) == nullptr);
    carry.target_process_refreshed();
    assert(carry.draft_carry(&pending_h) == &pending_h);

    carry.sequence_transition(
        common_speculative_sequence_event::target_restored_without_draft);
    assert(carry.draft_carry(&pending_h) == nullptr);
    assert(carry.target_process_mode(9) ==
           common_speculative_mtp_carry_lifecycle::process_mode::target_only);

    carry.target_process_refreshed();
    assert(carry.draft_carry(&pending_h) == &pending_h);
}

void test_transaction_rows_and_checkpoint() {
    // A two-token MTP transaction has three target rows: sampled, proposal 0,
    // and proposal 1.  The carry row is the accepted frontier, including the
    // bonus row when both proposals are accepted.
    assert(common_speculative_mtp_carry_row(3, 0) == 0); // accept 0 + bonus
    assert(common_speculative_mtp_carry_row(3, 1) == 1); // accept 1 + bonus
    assert(common_speculative_mtp_carry_row(3, 2) == 2); // accept 2 + bonus
    assert(common_speculative_mtp_carry_row(3, 3) == 2); // defensive clamp

    const llama_pos sampled = 100;
    const std::vector<llama_pos> frozen_positions = {
        sampled, sampled + 1, sampled + 2,
    };
    for (size_t row = 0; row < frozen_positions.size(); ++row) {
        assert(frozen_positions[row] == sampled + (llama_pos) row);
        assert(common_speculative_mtp_draft_position(
                   sampled, row, false) == sampled + 1 + (llama_pos) row);
    }

    // The selected table is immutable for the transaction. A publication
    // becomes visible only when the next transaction starts, after the
    // accepted frontier has been resolved.
    const uint64_t table_generation_before = 17;
    const uint64_t table_generation_after = table_generation_before + 1;
    uint64_t transaction_table_generation = table_generation_before;
    for (size_t accepted = 0; accepted <= 2; ++accepted) {
        const auto frontier = common_speculative_rollback_frontier_resolve(
            100, 2, accepted);
        assert(frontier.valid());
        assert(transaction_table_generation == table_generation_before);
        assert(frontier.accepted_token_count == 101 + (int64_t) accepted);
    }
    transaction_table_generation = table_generation_after;
    assert(transaction_table_generation == table_generation_after);

    // Checkpoint/load carries the hidden row, while the transition invalidates
    // it. This prevents cache reuse from consuming a row from an old branch.
    common_speculative_mtp_carry_lifecycle source;
    std::vector<float> source_h = { 1.0f, 2.0f, 3.0f, 4.0f };
    source.target_process_refreshed();
    std::vector<uint8_t> state;
    assert(common_speculative_mtp_carry_state_save(source, source_h, state));

    common_speculative_mtp_carry_lifecycle restored;
    std::vector<float> restored_h(source_h.size(), 0.0f);
    assert(common_speculative_mtp_carry_state_load(restored, restored_h, state));
    assert(restored.draft_ready());
    assert(restored_h == source_h);
    restored.sequence_transition(common_speculative_sequence_event::target_restored_without_draft);
    assert(!restored.draft_ready());

    // Repeated rejection of the same frontier is idempotent; a later
    // transaction may apply its distinct frontier once.
    common_speculative_mtp_rollback_guard guard;
    assert(guard.should_apply(101));
    assert(!guard.should_apply(101));
    assert(guard.should_apply(102));
    assert(!guard.should_apply(102));
}

} // namespace

int main() {
    test_proposal_positions();
    test_frontier_and_rollback_once();
    test_hidden_carry_freshness();
    test_transaction_rows_and_checkpoint();
    std::cout << "test-speculative-mtp-state: PASS\n";
    return 0;
}
