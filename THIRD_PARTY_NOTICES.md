# Third-party materials and notices

The root [MIT License](LICENSE) covers the original UrbanFlow AI software and
documentation. It does not replace the licenses, terms, attribution obligations,
or ownership notices of third-party material. Existing upstream notices must be
retained when redistributing their material.

## Taxi records and derived observation panels

The project uses historical NYC Yellow Taxi records and derived regional count
panels. The City of New York and the source providers retain their applicable
rights and terms. The project's MIT license does not relicense the underlying
source data. Source processing, filters, dataset identifiers and modifications
are described in the README and acquisition provenance.

- [NYC Open Data Terms of Use](https://data.cityofnewyork.us/stories/s/Terms-of-Use/k9k7-3cje)
- [NYC TLC trip-record data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

The source is historical data for this project's defined retained-trip target.
Neither the project nor its results imply endorsement by NYC Open Data or the
New York City Taxi and Limousine Commission.

## Software dependencies

Python and JavaScript dependencies retain their own upstream licenses and
copyright notices. The repository's MIT license is not a replacement for those
licenses. Consult the pinned dependency files, each package's distribution
metadata, and the license files shipped with installed packages when
redistributing an environment or application bundle.

## Notebook and report figures

Figures, embedded maps or other third-party material retained in historical
notebooks keep their original applicable notices and terms. Original analysis
code is covered by the root MIT License; that does not relabel source datasets
or third-party illustrations as MIT-licensed material.

## Historical README visuals

Images under `docs/legacy/` are preserved from the repository owner's earlier
README attachments. Their original canonical URLs, revisions and hashes are in
`docs/legacy/provenance.json`. Existing embedded map, tile or other third-party
attributions remain applicable. The figures are labeled as historical material;
the root MIT license does not relabel third-party material inside a screenshot.

## Historical weather and neighborhood area labels

Weather data is obtained from [Open-Meteo](https://open-meteo.com/en/docs/historical-weather-api),
using Copernicus ERA5 reanalysis; attribution: Open-Meteo and Copernicus Climate
Change Service/ECMWF. Open-Meteo data is provided under
[CC BY 4.0](https://open-meteo.com/en/licence). The public API's
[non-commercial usage terms](https://open-meteo.com/en/terms) apply.

Centroid area labels derive from NYC Department of City Planning's
[2020 Neighborhood Tabulation Areas](https://data.cityofnewyork.us/City-Government/2020-Neighborhood-Tabulation-Areas-NTAs-/9nt8-h7nd).
The data's NYC Open Data terms remain applicable. The project MIT license does
not replace provider data terms. Area labels do not define model boundaries.
