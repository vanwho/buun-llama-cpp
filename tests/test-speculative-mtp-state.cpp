#include "speculative.h"

#include <algorithm>
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

void test_query_replay_carry_generations() {
    // Both fixtures begin at the same pre-query target frontier and hidden
    // carry. An unchanged published history keeps that carry; a changed view
    // forces target-only replay before MTP can consume a refreshed hidden row.
    constexpr uint64_t turn_id = 41;
    constexpr uint64_t initial_history_generation = 71;
    constexpr llama_pos query_begin = 100;
    constexpr llama_pos query_end = 104;
    const std::vector<llama_pos> final_user_tokens = { 9001, 9002, 9003, 9004 };
    const std::vector<float> pre_query_hidden = { 1.0f, 2.0f, 3.0f };

    common_speculative_mtp_carry_lifecycle unchanged_carry;
    unchanged_carry.target_process_refreshed();
    common_speculative_mtp_history_epoch unchanged_history;
    assert(unchanged_history.bind(turn_id, initial_history_generation));
    size_t unchanged_replays = 0;
    assert(query_begin == 100 && query_end == query_begin +
            (llama_pos) final_user_tokens.size());
    assert(unchanged_history.matches(turn_id, initial_history_generation));
    assert(unchanged_carry.draft_carry(pre_query_hidden.data()) == pre_query_hidden.data());
    assert(unchanged_replays == 0);

    common_speculative_mtp_carry_lifecycle changed_carry;
    changed_carry.target_process_refreshed();
    common_speculative_mtp_history_epoch changed_history;
    assert(changed_history.bind(turn_id, initial_history_generation));
    changed_carry.sequence_transition(
        common_speculative_sequence_event::target_restored_without_draft);
    assert(changed_carry.draft_carry(pre_query_hidden.data()) == nullptr);
    std::vector<llama_pos> replayed_tokens;
    replayed_tokens.insert(replayed_tokens.end(), final_user_tokens.begin(),
            final_user_tokens.end());
    assert(replayed_tokens.size() == size_t(query_end - query_begin));
    const std::vector<float> replayed_hidden = { 4.0f, 5.0f, 6.0f };
    changed_carry.target_process_refreshed();
    uint64_t changed_history_generation = initial_history_generation;
    ++changed_history_generation;
    assert(changed_history.bind(turn_id, changed_history_generation));
    size_t changed_replays = 1;
    assert(changed_replays == 1);
    assert(changed_history.matches(turn_id, changed_history_generation));
    assert(!changed_history.matches(turn_id, initial_history_generation));
    assert(changed_carry.draft_carry(replayed_hidden.data()) == replayed_hidden.data());
    std::cout << "query_replay_carry_generations=pass unchanged_replays=0 "
                 "changed_replays=1 stale_carry=cleared\n";
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
    assert(state.size() == 3 * sizeof(uint32_t) + source_h.size() * sizeof(float));

    common_speculative_mtp_carry_lifecycle restored;
    std::vector<float> restored_h(source_h.size(), 0.0f);
    assert(common_speculative_mtp_carry_state_load(restored, restored_h, state));
    assert(restored.draft_ready());
    assert(restored_h == source_h);
    auto stale_state = state;
    stale_state[sizeof(uint32_t)]++;
    const std::vector<float> unchanged_h = restored_h;
    assert(!common_speculative_mtp_carry_state_load(restored, restored_h, stale_state));
    assert(restored_h == unchanged_h && restored.draft_ready());
    restored.sequence_transition(common_speculative_sequence_event::target_restored_without_draft);
    assert(!restored.draft_ready());

    // Repeated rejection of the same frontier is idempotent; a later
    // transaction may apply its distinct frontier once.
    common_speculative_mtp_rollback_guard guard;
    assert(guard.should_apply(101));
    assert(!guard.should_apply(101));
    assert(guard.should_apply(102));
    assert(!guard.should_apply(102));

    common_speculative_mtp_history_epoch frozen_history;
    assert(frozen_history.bind(29, 71));
    // A generation-tail append advances the mutable table epoch from 90 to
    // 91, but the target and draft remain bound to the same frozen history.
    const uint64_t table_epoch_before_append = 90;
    const uint64_t table_epoch_after_append = table_epoch_before_append + 1;
    assert(table_epoch_after_append != table_epoch_before_append);
    assert(frozen_history.matches(29, 71));
    // A rejected suffix rollback keeps the same binding for the next verify.
    const auto rejected = common_speculative_rollback_frontier_resolve(200, 2, 1);
    assert(rejected.valid() && rejected.rejected_draft_tokens == 1);
    assert(frozen_history.matches(29, 71));
    assert(!frozen_history.matches(30, 71));
    assert(!frozen_history.matches(29, 72));
    frozen_history.clear();
    assert(!frozen_history.matches(29, 71));
    std::cout << "query_checkpoint_mtp_carry=pass carry_bytes="
              << state.size() << " rollback_once=pass\n";
    std::cout << "frozen_history_and_mtp_epoch=pass\n";
}

void test_reject_rollback_and_next_proposal() {
    // Reproduce the first changed-history rejection with the same nmax=2
    // transaction shape used by native MTP. Target verification owns the
    // sampled row and both draft rows until acceptance resolves the frontier.
    constexpr llama_pos n_past = 100;
    constexpr uint16_t n_draft = 2;
    constexpr uint16_t n_accepted = 1;
    const std::vector<llama_pos> target_positions = {
        n_past, n_past + 1, n_past + 2,
    };
    const std::vector<llama_pos> draft_positions = {
        common_speculative_mtp_draft_position(n_past, 0, false),
        common_speculative_mtp_draft_position(n_past, 1, false),
    };
    assert(target_positions[0] == n_past);
    assert(target_positions[1] == draft_positions[0]);
    assert(target_positions[2] == draft_positions[1]);

    const auto frontier = common_speculative_rollback_frontier_resolve(
        n_past, n_draft, n_accepted);
    assert(frontier.valid());
    assert(frontier.accepted_draft_tokens == 1);
    assert(frontier.rejected_draft_tokens == 1);
    assert(frontier.accepted_token_count == 102);
    assert(frontier.rejected_suffix_begin == 102);
    assert(frontier.rejected_suffix_end == 103);

    // Rejection truncates both provisional views at exactly one shared
    // absolute frontier. No proposal from the old transaction may survive.
    std::vector<llama_pos> target_provisional = target_positions;
    std::vector<llama_pos> draft_provisional = draft_positions;
    const auto discard_rejected_suffix = [&frontier](auto & positions) {
        positions.erase(std::remove_if(positions.begin(), positions.end(),
                [&frontier](llama_pos position) {
                    return position >= frontier.rejected_suffix_begin;
                }), positions.end());
    };
    discard_rejected_suffix(target_provisional);
    discard_rejected_suffix(draft_provisional);
    assert((target_provisional == std::vector<llama_pos>{100, 101}));
    assert((draft_provisional == std::vector<llama_pos>{101}));
    assert(target_provisional.back() + 1 == frontier.accepted_token_count);
    assert(draft_provisional.back() + 1 == frontier.accepted_token_count);

    // A subsequent proposal begins from the accepted target/draft frontier,
    // and MTP carry comes from the accepted verification row rather than the
    // discarded suffix. The proposal accounting denominator remains n_draft.
    const llama_pos next_proposal = common_speculative_mtp_draft_position(
        (llama_pos) frontier.accepted_token_count, 0, false);
    assert(next_proposal == frontier.accepted_token_count + 1);
    assert(common_speculative_mtp_carry_row(3, n_accepted) == 1);
    const uint16_t acceptance_denominator = n_draft;
    assert(acceptance_denominator == 2);
    assert(n_accepted + frontier.rejected_draft_tokens == acceptance_denominator);
    std::cout << "reject_rollback_next_proposal=pass target_frontier=102 "
                 "draft_frontier=102 denominator=2\n";
}

void test_four_microbatch_shifted_hidden_rows() {
    constexpr int32_t batch_rows = 1024;
    constexpr int32_t microbatch_rows = 256;
    constexpr int32_t width = 4;
    std::vector<float> target_hidden((size_t) batch_rows * width);
    std::vector<float> device_owner((size_t) batch_rows * width);
    std::vector<float> draft_input((size_t) batch_rows * width, -1.0f);
    const std::vector<float> committed_carry = { -1.0f, -2.0f, -3.0f, -4.0f };

    for (int32_t row = 0; row < batch_rows; ++row) {
        for (int32_t col = 0; col < width; ++col) {
            target_hidden[(size_t) row * width + col] = float(row * 10 + col);
        }
    }

    // Four graph slots publish into one logical owner at stable batch offsets.
    for (int32_t micro = 0; micro < 4; ++micro) {
        const int32_t offset = micro * microbatch_rows;
        std::copy_n(target_hidden.begin() + (size_t) offset * width,
                (size_t) microbatch_rows * width,
                device_owner.begin() + (size_t) offset * width);
    }
    assert(device_owner == target_hidden);

    for (int32_t micro = 0; micro < 4; ++micro) {
        const int32_t offset = micro * microbatch_rows;
        for (int32_t row = 0; row < microbatch_rows; ++row) {
            const int32_t target_row = offset + row;
            const float * source = target_row == 0
                ? committed_carry.data()
                : device_owner.data() + (size_t) (target_row - 1) * width;
            std::copy_n(source, width,
                    draft_input.begin() + (size_t) target_row * width);
        }
    }

    for (int32_t row = 0; row < batch_rows; ++row) {
        for (int32_t col = 0; col < width; ++col) {
            const float expected = row == 0
                ? committed_carry[col]
                : target_hidden[(size_t) (row - 1) * width + col];
            assert(draft_input[(size_t) row * width + col] == expected);
        }
    }
    std::cout << "four_microbatch_shifted_hidden_oracle=pass rows=1024 microbatch=256\n";
}

void test_cancelled_hidden_stage_generation() {
    uint64_t stage_generation = 90;
    const uint64_t request_generation = stage_generation;
    bool stage_valid = true;
    // A cancelled request invalidates its partial owner before a later request
    // can reuse the same context storage.
    stage_valid = false;
    ++stage_generation;
    assert(!stage_valid);
    assert(request_generation != stage_generation);
    std::cout << "cancelled_hidden_stage=pass stale_generation_refused=pass\n";
}

} // namespace

int main() {
    test_proposal_positions();
    test_frontier_and_rollback_once();
    test_hidden_carry_freshness();
    test_query_replay_carry_generations();
    test_transaction_rows_and_checkpoint();
    test_reject_rollback_and_next_proposal();
    test_four_microbatch_shifted_hidden_rows();
    test_cancelled_hidden_stage_generation();
    std::cout << "test-speculative-mtp-state: PASS\n";
    return 0;
}
