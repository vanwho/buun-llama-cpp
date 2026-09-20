# 85d — remove host-bound selected-path maintenance

Revision: `hotpath-v10-20260914`. Task: `85-04`.

Use the existing pager host worker, mailbox, and event ownership to coalesce
maintenance. A 256-token page is sealed once it is complete; scheduler fences
must not rebuild summaries, publish routing tables, or rebuild graphs per
U-token when refresh is not due.
