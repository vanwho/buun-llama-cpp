# 53a-vbr-scratch-v10

Revision: `hotpath-v10-20260914`. Task: `53-01`.

Repair the measured full-L request failure at the VBR f16 dequant scratch
boundary. Keep canonical Turbo4 KV, full-L GPU native-MTP, graph-capture
safety, and recoverable pre-request allocation failure. The smallest fixture is
`L262144/H30208/B128/U128` with a 1200-token request; do not start another
occupancy campaign until this boundary is proven.

