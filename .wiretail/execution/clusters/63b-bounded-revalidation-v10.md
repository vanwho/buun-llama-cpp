# 63b — bounded post-repair V10 revalidation

Revision: `hotpath-v10-20260914`. Task: `63-02`.

Run the canonical bounded smoke first, then only the phase-61 rows admitted by
that smoke. Preserve failed and not-run rows and never infer occupancy,
promotion, quality, speed, or MTP acceptance from allocation or startup.
