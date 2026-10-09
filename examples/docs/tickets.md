# Semantic search over support tickets

Semantic search maps every ticket to an embedding vector and answers
nearest-neighbor queries in milliseconds. Dense retrieval understands
paraphrase: a ticket saying "my card was charged twice" matches "duplicate
billing on my credit card" even though they share almost no words.

The failure mode is exact terminology. A query for the literal error code
ERR_BILLING_402 may retrieve tickets about billing in general instead of the
runbook containing that exact string. Hybrid retrieval fixes this by running
keyword search alongside vector search and merging the ranked lists.
