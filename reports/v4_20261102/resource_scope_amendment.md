# Resource scope amendment before historical-model outcomes/final access

Observed historical API calls take roughly 1–3 minutes per day, making a full
two-month raw acquisition expensive relative to the hypothesis. The selected
scope is November 2–30 (29 days) plus December 10–12, 17–19 and 24–26 (nine days).
The December dates provide Christmas and adjacent-day outcomes with matched
same-weekday references one/two weeks earlier. They were selected by calendar
and resource cost, not observed forecasting errors. The primary June and
secondary July protocol are unchanged.

Unacquired December days are **absent**, never zero demand. Feature training
requires a full 16-day consecutive elapsed-time history and cannot cross a gap.
Thus the 2015 boosted-model labels come from the valid November segment; the
Christmas snapshot days contribute only to train-only normalized holiday
statistics. January–May is the unchanged continuous 2016 panel. This is a small
historical-coverage test, not a full annual-seasonality study.

Bulk acquisition uses at most two concurrent daily requests. Two bounded
aggregate feasibility probes were also made (one later probe ran alongside the
bulk requests); the latter matched an existing day exactly but took over two
minutes, so bulk processing was not replaced without a demonstrated efficiency
benefit. No final target was used in those probes. CSV round-trip float parsing
reduces memory/response overhead while leaving numeric coordinates and mapping
unchanged. Completed JSON/CSV-derived daily aggregates are reused, not replaced.
