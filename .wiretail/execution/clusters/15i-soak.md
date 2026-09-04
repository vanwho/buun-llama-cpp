# Cluster 15i — corrective endurance and transition to handoff

Tasks: `15-09`.

Purpose: soak the accepted selective+MTP profile through churn, focus shifts, rollback, cancellation,
checkpoint restore, restart, and handoff without leaks. Keep the accepted profile loaded on success for
phase 16; restore the original profile only if explicitly requested by the final cleanup policy.

Exit artifact: V3 soak manifest and a ready phase-16 handoff state.
