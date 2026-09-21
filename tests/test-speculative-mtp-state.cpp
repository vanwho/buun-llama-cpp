#include "speculative.h"

#include <cassert>
#include <iostream>

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

} // namespace

int main() {
    test_proposal_positions();
    test_frontier_and_rollback_once();
    test_hidden_carry_freshness();
    std::cout << "test-speculative-mtp-state: PASS\n";
    return 0;
}
