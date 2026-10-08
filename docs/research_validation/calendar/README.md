# Scheduled exchange sessions for the protocol freeze

`xnys_scheduled_sessions_20261002_20271008.csv` reconstructs the published NYSE
core-equity schedule from its [holiday/hour table](https://www.nyse.com/trade/hours-calendars)
and the [ICE/NYSE calendar announcement](https://ir.theice.com/press/news-details/2025/NYSE-Group-Announces-2026-2027-and-2028-Holiday-and-Early-Closings-Calendar/default.aspx).
Verification and file hash are in `metadata.json`.

The October 2, 2026 session had already opened when this was prepared. If final
code and protocol are frozen before the next session, the first complete
scheduled holdout session starts **October 5, 2026, 9:30 a.m. New York time**.
The freeze code selects from timestamped sessions, not this prose. It must not
backdate the holdout when a final freeze happens later.

Conditional six-/twelve-month review checkpoints are April 5 and October 5,
2027, after their scheduled closes. These dates do not establish statistical
power, schedule an automation, or authorize trading. Extraordinary closures
may alter the announced schedule; recheck official notices before operation.

No third-party calendar package was installed. The CSV only covers the stated
range and includes daylight-saving timezone changes and official early closes.
