# SPEED25_09_DECODE

The paged Turbo4 dispatcher now selects query tiles 1/2/4/8/16/32/64 from
the active query count when no explicit tile is supplied. Shared memory is
sized from that active tile, and CUDA device properties and the largest
configured dynamic-shared attribute are cached per device. Existing explicit
tiles and unsupported-shape fallback remain available.

The RTX 4080 fixture passed all existing correctness checks. For 530 selected
rows, default Q=1/2/3/5 measured 0.517/0.532/0.605/1.066 ms. The large-query
sweep measured tile 16 faster than 32 and 64 at Q=64 and Q=256; therefore the
implementation does not claim that the largest tile is universally best.
Split capacity 3 remained faster than serial at Q=512 (20.159 vs 21.804 ms)
on this fixture.

Raw output: `raw/SPEED25_09_DECODE_cuda.txt`.
