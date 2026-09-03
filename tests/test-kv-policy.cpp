#include "llama-kv-policy.h"

#include <algorithm>
#include <cassert>
#include <iostream>

static llama_kv_policy_page page(uint64_t id, bool resident = true) {
    llama_kv_policy_page p; p.id = id; p.resident = resident; p.age = id; p.recency = id; return p;
}

static bool contains(const std::vector<uint64_t> & values, uint64_t id) {
    return std::find(values.begin(), values.end(), id) != values.end();
}

int main() {
    llama_kv_policy_trace trace;
    trace.epoch = 1; trace.capacity_pages = 2; trace.write_page = 4;
    auto p1 = page(1); p1.attention_ema_q = 20; p1.attention_observed = true;
    auto p2 = page(2); p2.attention_ema_q = 1; p2.attention_observed = true; p2.dirty_cost = 9;
    auto p3 = page(3); p3.application_pin = true;
    auto p4 = page(4, false); p4.anchor = true;
    auto p5 = page(5); // cold: must not be interpreted as observed attention == zero
    trace.pages = { p1, p2, p3, p4, p5 };
    trace.summary_top_k = { 5, 5 };
    trace.exploration = { 2 };
    const auto result = llama_kv_policy_replay(trace);
    assert(result.status == llama_kv_policy_status::ok);
    assert(result.retrieve.size() == 5);
    assert(result.victims.size() == 2);
    assert(result.victims[0] == 2); // observed low attention wins despite dirty cost
    assert(result.victims[1] == 5); // cold page follows explicit fallback, not zero attention

    trace.capacity_pages = 1;
    assert(llama_kv_policy_replay(trace).status == llama_kv_policy_status::pin_overflow);
    trace.capacity_pages = 3; trace.pages[0].application_pin = true;
    trace.pages[1].application_pin = true;
    trace.pages[2].inflight_pin = true;
    trace.pages[4].application_pin = true;
    assert(llama_kv_policy_replay(trace).status == llama_kv_policy_status::pin_overflow);

    llama_kv_policy_trace controller_trace;
    controller_trace.epoch = 7;
    controller_trace.write_page = 1;
    controller_trace.pages.resize(8);
    for (size_t i = 0; i < controller_trace.pages.size(); ++i) {
        controller_trace.pages[i] = page(i + 1, i < 6);
        controller_trace.pages[i].age = 8 - i;
        controller_trace.pages[i].recency = i;
        controller_trace.pages[i].attention_ema_q = i * 10;
    }
    controller_trace.pages[0].current = true;
    controller_trace.pages[1].structural = true;
    controller_trace.pages[2].recent = true;
    controller_trace.pages[3].attention_observed = true;
    controller_trace.pages[4].attention_observed = true;
    controller_trace.pages[5].speculative_pin = true;
    controller_trace.summary_top_k = { 7, 4, 7 };
    controller_trace.exploration = { 8, 4 };

    llama_kv_policy_controller_config controller_config;
    controller_config.capacity_pages = 6;
    controller_config.recent_pages = 1;
    controller_config.structural_pages = 1;
    controller_config.historical_pages = 2;
    controller_config.transient_pages = 2;
    const auto decision = llama_kv_policy_decide(controller_trace, controller_config);
    assert(decision.status == llama_kv_policy_status::ok);
    assert(decision.target.size() == controller_config.capacity_pages);
    assert(contains(decision.target, 1));
    assert(contains(decision.target, 6));
    assert(decision.unavailable_evidence == 6);
    const auto repeat = llama_kv_policy_decide(controller_trace, controller_config);
    assert(decision.target == repeat.target);
    const auto retained = llama_kv_policy_decide(controller_trace, controller_config, decision.target);
    assert(retained.keeps.size() == decision.target.size());
    controller_config.capacity_pages = 1;
    assert(llama_kv_policy_decide(controller_trace, controller_config).status == llama_kv_policy_status::pin_overflow);

    std::cout << "kv policy checks passed\n";
}
