
## Main Files

- `report.pdf`: final report.
- `data/processed/market_day_panel.csv`: complete dataset used by the regressions.
- `data/raw_sample/raw_market_sample_first100.json`: readable sample of 100 raw market records.
- `data/raw_sample/raw_market_sample_first100.jsonl`: the same sample in source JSONL format.
- `DATA_DESCRIPTION.md`: sources, sample selection, transformations, and limitations.
- `DATA_DICTIONARY.md`: definitions of raw and processed variables.

## Empirical Sample

The analysis uses 3,234 finalized binary Kalshi contracts satisfying:

```text
total contract volume >= 100,000
contract duration >= 30 days
final result in {yes, no}
```

Transaction records are aggregated into an unbalanced contract-day panel containing
246,577 observations, 3,234 contracts, and 1,147 events. A contract-day enters the
panel when at least one transaction occurs.


The raw transaction archive contains more than 100GB 13 million records. I put
100-record raw market sample demonstrates the original JSON structure. The complete panel CSV is included and is sufficient to reproduce every table,
figure, simulation in the report.
