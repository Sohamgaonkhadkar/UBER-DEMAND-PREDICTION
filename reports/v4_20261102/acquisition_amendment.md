# Acquisition feasibility amendment, before model fitting and final access

The server-side fixed-Voronoi aggregation probe on the already-spent May 16
region-2 data timed out after 120 seconds. It is not used to label any record.
The fallback uses the previously verified local nearest-centroid method.

Historical acquisition is bounded to November 2–December 31, 2015 (60 days),
rather than four 2015 months. This adds Thanksgiving and Christmas while
reducing network/compute needs. November 1 is omitted because local source
timestamps do not disambiguate the repeated fall-DST hour; no invented fold or
zero-filled unknown hour enters training. November 2 onward joins the existing
January–May 2016 panel into a continuous elapsed-time series. This choice uses
known calendar/source metadata, not forecast outcomes.

Official 2015 metadata uses `pickup_datetime`, whereas 2016 uses
`tpep_pickup_datetime`; the query explicitly aliases the former after applying
identical bounds/fare/distance filters. Failed 400 responses yielded no accepted
data. Final June and July windows, model selection and metric are unchanged.
