# Cluster 15c — live harness and persistent server lifecycle

Tasks: `15-03`.

Purpose: turn the adapter into a real, explicit Qwen3.8 benchmark launcher and define stateful service
lifecycle. Capture the starting profile, but leave a successfully launched test profile loaded for the
next task. Restore only when a packet explicitly requests a control/revert benchmark, teardown, failure
recovery, or final cleanup. Never touch the unrelated 8092 service.

Exit artifact: reproducible launcher, metrics/placement capture, lifecycle state file, and isolated smoke.
