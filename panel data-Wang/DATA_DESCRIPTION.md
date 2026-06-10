# Data Description

## 1. Data Source

The project uses publicly accessible Kalshi historical market metadata and transaction
records. Kalshi binary contracts pay one dollar when the specified event occurs and
zero otherwise. The Yes-side transaction price can therefore be interpreted as a
market-implied event probability.

The raw data have two levels:

1. **Market metadata:** one JSON record per contract, containing identifiers, opening
   and closing times, contract rules, total volume, and the final settlement result.
2. **Transaction records:** one JSON record per trade, containing the contract ticker,
   transaction timestamp, Yes and No prices, and transaction size.

The file `data/raw_sample/raw_market_sample_first100.json` contains 100 representative
raw market-metadata records. The JSONL version preserves the original one-record-per-line
format.

## 2. Contract Selection

The empirical sample retains finalized binary contracts satisfying all of the following:

- `volume_fp >= 100000`
- `duration_days >= 30`
- `result` equals `yes` or `no`

Contract duration is measured as the elapsed time between `open_time` and `close_time`.
The restrictions focus the analysis on actively traded, sufficiently long-lived
contracts with observed binary settlements. Records with `result = scalar` are
excluded because they settle using a continuous numerical value and therefore cannot
be represented by the binary outcome variable used in the Bernoulli models.

The selected market-level sample contains:

| Statistic | Value |
|---|---:|
| Finalized binary contracts | 3,234 |
| Final Yes outcomes | 979 |
| Final No outcomes | 2,255 |
| Minimum total volume | 100,000 |
| Minimum duration | 30 days |

## 3. Construction of the Contract-Day Panel

Transactions are grouped by contract ticker and UTC calendar date. For each
contract-day:

- `last_price` is the Yes price from the final transaction observed that day.
- `daily_contract_volume` is the sum of transaction sizes during the day.
- `daily_trade_count` is the number of transactions during the day.
- `outcome` equals one for a final Yes settlement and zero for a final No settlement.
- `absolute_forecast_error` is calculated as `abs(outcome - last_price)`.
- `time_to_close_days` is the number of days from the observation date to contract close.

Only dates with at least one observed trade enter the panel. Consequently, the data form
an unbalanced trading-day panel rather than a complete calendar-day panel.

## 4. Processed Dataset

The complete regression dataset is:

```text
data/processed/market_day_panel.csv
```

Its dimensions are:

| Statistic | Value |
|---|---:|
| Contract-day observations | 246,577 |
| Unique contracts | 3,234 |
| Unique events | 1,147 |
| Earliest observation date | 2021-07-02 |
| Latest observation date | 2026-04-01 |
| Median observed trading days per contract | 48 |
| Minimum observed trading days per contract | 4 |
| Maximum observed trading days per contract | 510 |

## 5. How the Data Enter the Models

The main linear panel dependent variable is:

\[
AFE_{it}=|Y_i-p_{it}|.
\]

The principal explanatory variables are:

- time to close, scaled in 30-day units;
- `log(1 + daily_contract_volume)`;
- `log(1 + daily_trade_count)`.

The contract ticker defines the panel unit. Event tickers are used to cluster standard
errors because multiple contracts can belong to the same event.

The Bernoulli GEE uses the final binary settlement outcome together with all observed
daily prices, time to close, and activity variables. Fixed-horizon calibration checks
use one observation per contract at approximately 30, 14, 7, 3, and 1 days before close.

## 6. Limitations

- Non-trading days are excluded because no new transaction price is observed.
- Contracts with more observed trading days receive more weight in the all-day GEE and
  calibration figure.
- Trading activity may respond to uncertainty or news, so activity coefficients are
  interpreted as conditional associations rather than causal effects.
- The high-volume, long-duration selection criteria mean that the sample does not
  represent every Kalshi contract.
