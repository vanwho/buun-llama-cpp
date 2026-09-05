# Cluster 17h — full-context quality, speed, and pressured soak

Tasks `17-15`–`17-17` rerun model-backed quality, paired performance, and
lifecycle soak at contract-valid contexts with physical pager pressure. Every
target and native draft KV path is Turbo4, and performance repeats the original
three-prompt protocol. Every run uses one binary/model/tokenizer/corpus
provenance and records nulls when a required comparison remains unavailable.
Before any long matrix, 17-15 must pass a bounded stable-startup gate at the
resolved context. The prior pressure run SIGSEGVed after listening; capture or
isolate that crash (or emit a deterministic refusal) and quiesce any
`Restart=always` loop before restoration rather than consuming the full
benchmark budget on an unstable service.
