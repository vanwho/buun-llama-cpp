# repair88 - occupied frontier scaling

Revision: `hotpath-v10-20260914`. Amendment: `repair87-20260921`.

This cluster advances the cache-preserving occupied frontier only after
reliability and matched-speed gates. It records full-L allocation, resident
occupancy, committed C, and prefill/draft/verify/rollback high-water values
independently.
