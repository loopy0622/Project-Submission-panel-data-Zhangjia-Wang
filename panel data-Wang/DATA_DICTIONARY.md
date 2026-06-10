# Data Dictionary

## Raw Market Sample

The raw sample files contain the original Kalshi market-metadata structure. Important
fields include:

| Field | Description |
|---|---|
| `ticker` | Unique contract identifier. |
| `event_ticker` | Identifier shared by contracts belonging to the same event. |
| `title` | Human-readable contract question. |
| `open_time` | Timestamp when the contract opened. |
| `close_time` | Timestamp when the contract closed. |
| `result` | Final settlement result. Binary contracts settle as `yes` or `no`; `scalar` indicates a contract settled using a continuous numerical value rather than a binary outcome. Scalar contracts are excluded from this project. |
| `volume_fp` | Total number of contracts traded over the contract life. |
| `last_price_dollars` | Final observed Yes-side market price in dollars. |
| `strike_type` | Structure of the settlement condition, such as `greater`, `less`, `between`, `custom`, or `structured`. |
| `rules_primary` | Primary settlement rule. |
| `updated_time` | Timestamp recorded by Kalshi for the latest change to the source market record; it is an original metadata field, not a project-processing timestamp. |
| `duration_days` | Constructed number of days between open and close. |

The raw JSON contains additional market-design, quote, settlement, and rule fields that
are retained for transparency but are not all used in the empirical regressions.

## Processed Market-Day Panel

| Variable | Type | Description | Used in analysis |
|---|---|---|---|
| `ticker` | string | Contract identifier and panel unit. | Yes |
| `event_ticker` | string | Event identifier used for event-clustered standard errors. | Yes |
| `date` | date | UTC trading date. | Yes |
| `outcome` | integer | Final settlement: Yes = 1, No = 0. | Yes |
| `last_price` | numeric | Final Yes-side transaction price observed that day. | Yes |
| `absolute_forecast_error` | numeric | `abs(outcome - last_price)`. | Yes |
| `time_to_close_days` | numeric | Days remaining until contract close. | Yes |
| `daily_contract_volume` | numeric | Sum of quantities traded for contract \(i\) on day \(t\). | Yes |
| `daily_trade_count` | integer | Number of transactions observed for contract \(i\) on day \(t\). | Yes |
| `total_contract_volume` | numeric | Total lifetime volume of the contract. | Descriptive/sample selection |
| `duration_days` | numeric | Contract lifetime in days. | Descriptive/sample selection |
| `strike_type` | string | Contract settlement-condition structure. | Retained but not used |

## Constructed Regression Variables

The analysis script constructs:

| Variable | Definition |
|---|---|
| `time_to_close_30d` | `time_to_close_days / 30` |
| `log_daily_volume` | `log(1 + daily_contract_volume)` |
| `log_trade_count` | `log(1 + daily_trade_count)` |
| `lag_afe` | Previous observed trading-day AFE within the same contract |
| `logit_price` | `log(last_price / (1 - last_price))` |
| `price_time_interaction` | `logit_price * time_to_close_30d` |
