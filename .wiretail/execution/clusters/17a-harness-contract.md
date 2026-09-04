# Cluster 17a — benchmark lifecycle and telemetry contract

Tasks `17-01`–`17-03` repair the boundary between the canonical profile runner,
the pager adapter, live service identity, and exported metrics. Keep the work
portable in the Buun repository; `/srv/ai` values are runtime inputs only.

Read the repository instructions, the three task packets, and the 15-03/16-03
handoffs. Do not load the giant historical 16-03 narrative. Preserve
keep-loaded-on-success, but restore any failed or post-validation-rejected run.
The cluster gate is an authenticated candidate smoke with complete telemetry,
correct MTP identity, and verified cleanup semantics.
