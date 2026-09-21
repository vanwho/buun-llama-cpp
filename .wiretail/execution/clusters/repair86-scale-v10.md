# repair86 - context scaling capability

Revision: `hotpath-v10-20260914`. Amendment: `repair86-20260921`.

Run the bounded scaling ladder only after the long-context prefill invariant is
repaired. Keep full-L Turbo4 native GPU draft placement and record allocation,
legal output reserve, committed C, high-water memory, and stop reasons
separately for 32K, 128K, and 256K. Do not infer occupancy from startup.
