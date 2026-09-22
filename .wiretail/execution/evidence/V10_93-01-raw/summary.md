# Bounded MTP diagnostic (93-01)

- status: `setup_failure`
- first_bad_rung: `None`
- setup_failure: `{'rung': 'mtp_off_dense_all_gpu', 'errors': ['<urlopen error [Errno 111] Connection refused>']}`

Acceptance is scored only from request-local fields. Reference, CPU, non-Turbo4, or identity-mismatched routes are setup/evidence failures; no speed or 256K claim is made.

| rung | status | requests |
|---|---|---:|
| `mtp_off_dense_all_gpu` | `setup_failure` | 0 |
| `mtp_on_dense_all_gpu` | `setup_failure` | 0 |
| `mtp_on_selected_paged_resident` | `setup_failure` | 0 |
| `mtp_on_selected_paged_cold_probe` | `setup_failure` | 0 |
