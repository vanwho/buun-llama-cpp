# V10 cold-promotion proof

Task `51-01` was exercised against the rebuilt candidate on an isolated GPU
server at `127.0.0.1:18080`. The successful candidate remains loaded after the
proof run. No client or server force-candidate API was used.

| Proof | Result | Evidence |
| --- | --- | --- |
| `controlled_model_query_promotion` | pass | Real model fixture, selected-packed route, 32 logical/16 physical pages, 82 promotions, 7872 H2D submissions/completions, and a complete selector → H2D → mapping → target-use edge. Forced page is `-1`. |
| `organic_chain_observation` | pass | T2 old-topic query selected cold logical page 15 at rank 0; host-ready, H2D completed, mapping published, and target graph consumed it. |
| `file_roundtrip_promotion` | pass | Fresh T3 A/B/A-again run; before A-again there were 16 resident and 15 host pages, and B produced the same complete cold-page edge. |
| `useful_answer` | false | The API content field was empty because reasoning-preserve consumed the 32-token completion budget. This is reported separately and is not used to weaken the mechanics proofs. |

The organic and file-roundtrip edge tables use the same stable identity:
logical page `15`, page/content generation `104`, physical slot `15`, and one
256-token H2D transfer of `4325376` useful and aligned bytes. Their completed
target-use epochs are `18005` and `16022`, respectively.

The fresh T3 documents were tokenizer-measured at 3462 tokens for A, 6870 for
B, and 6903 for A-again. The measured state before A-again was 16 resident,
15 host, 32 logical pages, and 8 selected attention pages (2048 tokens), so A
was verified host-backed rather than inferred cold from distance alone.

Raw logs and request/slot snapshots are listed in `V10_COLD_PROOF.json`.
