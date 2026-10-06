"""The page's regions as lists of countries under ECDC's names (countriesAndTerritories in ecdc.csv).

They are plot_series.py's region lists of 2020, which drew the page's regional line charts and heat maps, with the names it spelt differently from ECDC corrected: 'Czech Republic' is Czechia (the 2020 EU charts left it out because of this), 'Guinea-Bissau' Guinea_Bissau and 'São Tomé and Príncipe' Sao_Tome_and_Principe. Dropped, because ECDC has no rows for them: Netherlands Antilles (dissolved in 2010) and Turkmenistan (which reported no cases). plot_series.py's duplicate spellings ('Cote dIvoir', 'Swaziland') are gone.

Added in October 2026, because the 2020 lists missed them although ECDC reports them, so the regional charts left them out without saying so: Western Sahara (North Africa); Syria and Yemen (Western Asia); Tajikistan (Central Asia); Laos and Timor-Leste (South-East Asia); the Falkland Islands (South America); Anguilla, Aruba, Bonaire, Saint Eustatius and Saba, the British Virgin Islands, Curaçao, Dominica, Grenada, Montserrat, Saint Kitts and Nevis, Sint Maarten, the Turks and Caicos Islands and the US Virgin Islands (Caribbean); the Marshall Islands, Micronesia, New Caledonia, the Northern Mariana Islands and Vanuatu (Oceania). Wallis and Futuna stays out: ECDC gives it no population, so no rate per million can be drawn for it.

Run it to check every name against ecdc.csv, and that every place ECDC reports is in a region: python3 regions.py
"""

from pathlib import Path

PARTS: dict[str, list[str]] = {
    'EU': [
        'Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czechia', 'Denmark', 'Estonia', 'Finland',
        'France', 'Germany', 'Greece', 'Hungary', 'Ireland', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg',
        'Malta', 'Netherlands', 'Poland', 'Portugal', 'Romania', 'Slovakia', 'Slovenia', 'Spain', 'Sweden',
    ],
    'EuropeNorth': [
        'Denmark', 'Sweden', 'Norway', 'Iceland', 'Finland', 'Greenland', 'Faroe_Islands', 'Estonia', 'Latvia',
        'Lithuania',
    ],
    'EuropeSouth': [
        'Greece', 'Italy', 'Malta', 'Portugal', 'Spain', 'France', 'Monaco', 'Holy_See', 'San_Marino', 'Gibraltar',
    ],
    'EuropeWest': [
        'Austria', 'Belgium', 'Czechia', 'France', 'Germany', 'Ireland', 'Liechtenstein', 'Luxembourg', 'Monaco',
        'Netherlands', 'Switzerland', 'United_Kingdom', 'Andorra', 'Jersey', 'Guernsey', 'Isle_of_Man',
    ],
    'EuropeEastCentral': [
        'Croatia', 'Albania', 'Armenia', 'Azerbaijan', 'Belarus', 'Bosnia_and_Herzegovina', 'Latvia', 'Lithuania',
        'Georgia', 'Moldova', 'Russia', 'Ukraine', 'Serbia', 'Kosovo', 'Montenegro', 'North_Macedonia', 'Slovenia',
    ],
    'AmericaNorth': ['United_States_of_America', 'Mexico', 'Canada', 'Bermuda'],
    'AmericaCentral': ['Belize', 'Costa_Rica', 'El_Salvador', 'Honduras', 'Guatemala', 'Panama', 'Nicaragua'],
    'AmericaSouth': [
        'Brazil', 'Colombia', 'Argentina', 'Peru', 'Venezuela', 'Chile', 'Ecuador', 'Bolivia', 'Paraguay', 'Uruguay',
        'Guyana', 'Suriname', 'Falkland_Islands_(Malvinas)',
    ],
    'Caribbean': [
        'Bahamas', 'Cayman_Islands', 'Cuba', 'Haiti', 'Dominican_Republic', 'Jamaica', 'Puerto_Rico',
        'Antigua_and_Barbuda', 'Trinidad_and_Tobago', 'Saint_Vincent_and_the_Grenadines', 'Barbados', 'Saint_Lucia',
        'Anguilla', 'Aruba', 'Bonaire, Saint Eustatius and Saba', 'British_Virgin_Islands', 'Curaçao', 'Dominica',
        'Grenada', 'Montserrat', 'Saint_Kitts_and_Nevis', 'Sint_Maarten', 'Turks_and_Caicos_islands',
        'United_States_Virgin_Islands',
    ],
    'AsiaSouthEast': [
        'Indonesia', 'Thailand', 'Philippines', 'Malaysia', 'Singapore', 'Vietnam', 'Cambodia', 'Brunei_Darussalam',
        'Myanmar', 'Laos', 'Timor_Leste',
    ],
    'AsiaCentral': ['Afghanistan', 'Kazakhstan', 'Uzbekistan', 'Kyrgyzstan', 'Tajikistan'],
    'AsiaEast': ['China', 'Japan', 'Mongolia', 'South_Korea', 'Taiwan'],
    'AsiaSouth': ['India', 'Pakistan', 'Afghanistan', 'Bangladesh', 'Nepal', 'Sri_Lanka', 'Bhutan', 'Maldives'],
    'AsiaWestern': [
        'Armenia', 'Azerbaijan', 'Bahrain', 'Egypt', 'Qatar', 'Kuwait', 'Oman', 'United_Arab_Emirates',
        'Saudi_Arabia', 'Israel', 'Iran', 'Iraq', 'Georgia', 'Turkey', 'Lebanon', 'Jordan', 'Palestine', 'Syria',
        'Yemen',
    ],
    'AfricaNorth': ['Algeria', 'Egypt', 'Morocco', 'Libya', 'Tunisia', 'Western_Sahara'],
    'AfricaEast': [
        'Djibouti', 'Eritrea', 'Ethiopia', 'Somalia', 'Sudan', 'South_Sudan', 'Madagascar', 'Mauritius', 'Comoros',
        'Seychelles', 'Uganda', 'Rwanda', 'Burundi', 'Kenya', 'United_Republic_of_Tanzania', 'Mozambique', 'Malawi',
        'Zambia', 'Zimbabwe',
    ],
    'AfricaCentral': [
        'Angola', 'Cameroon', 'Central_African_Republic', 'Chad', 'Democratic_Republic_of_the_Congo', 'Congo',
        'Equatorial_Guinea', 'Gabon', 'Sao_Tome_and_Principe',
    ],
    'AfricaSouth': ['Botswana', 'Eswatini', 'Lesotho', 'Namibia', 'South_Africa'],
    'AfricaWest': [
        'Benin', 'Burkina_Faso', 'Cape_Verde', 'Cote_dIvoire', 'Gambia', 'Ghana', 'Guinea', 'Guinea_Bissau',
        'Liberia', 'Mali', 'Mauritania', 'Niger', 'Nigeria', 'Senegal', 'Sierra_Leone', 'Togo',
    ],
    'Oceania': [
        'Australia', 'Papua_New_Guinea', 'New_Zealand', 'Fiji', 'French_Polynesia', 'Guam', 'Solomon_Islands',
        'Marshall_Islands', 'Micronesia_(Federated_States_of)', 'New_Caledonia', 'Northern_Mariana_Islands', 'Vanuatu',
    ],
    'Nordic': ['Denmark', 'Sweden', 'Norway', 'Finland', 'Iceland', 'Greenland', 'Faroe_Islands'],
}


def union(*names: str) -> list[str]:
    """The countries of several parts, each once, in alphabetical order."""
    return sorted({country for name in names for country in PARTS[name]})


# The regions that have a section on the page, as plot_series.py composed them.
REGIONS: dict[str, list[str]] = {
    'EU': sorted(PARTS['EU']),
    'Europe': union('EU', 'EuropeEastCentral', 'EuropeNorth', 'EuropeSouth', 'EuropeWest'),
    'Americas': union('AmericaSouth', 'AmericaNorth', 'AmericaCentral', 'Caribbean'),
    'Asia': union('AsiaSouthEast', 'AsiaCentral', 'AsiaEast', 'AsiaSouth', 'AsiaWestern'),
    'Africa': union('AfricaNorth', 'AfricaEast', 'AfricaSouth', 'AfricaWest', 'AfricaCentral'),
    'Oceania': sorted(PARTS['Oceania']),
    'Nordic': sorted(PARTS['Nordic']),
}


# ECDC rows that belong to no region on purpose.
OUTSIDE = {'Wallis_and_Futuna', 'Cases_on_an_international_conveyance_Japan'}


def main() -> None:
    import pandas as pd

    ecdc = set(pd.read_csv(Path(__file__).resolve().parent / 'ecdc.csv', usecols=['countriesAndTerritories'])
               ['countriesAndTerritories'])
    listed = {country for countries in PARTS.values() for country in countries}
    unknown = sorted(listed - ecdc)
    unplaced = sorted(ecdc - listed - OUTSIDE)
    for region, countries in REGIONS.items():
        print(f'{region}: {len(countries)} countries and territories')
    if unknown:
        raise SystemExit(f'Not in ecdc.csv: {unknown}')
    if unplaced:
        raise SystemExit(f'In ecdc.csv but in no region: {unplaced}')
    print('Every name is in ecdc.csv, and every place in ecdc.csv is in a region.')


if __name__ == '__main__':
    main()
