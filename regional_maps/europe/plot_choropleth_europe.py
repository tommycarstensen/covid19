import os
import zipfile

import geopandas as gpd
import matplotlib.pyplot as plt

# import shapefile
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
import pandas as pd
import requests
from geopandas import GeoDataFrame
from matplotlib.colors import Normalize

# Please note that daily data on cases per subnational region are not available for all countries. Weekly data on new cases per subnational region for all EU/EEA countries and the UK can be found at 'Download data on the weekly subnational 14-day notification rate of new COVID-19 cases'. There may be differences between the rates shown in these two datasets since they are based on different sources of data.
# https://www.ecdc.europa.eu/en/publications-data/subnational-14-day-notification-rate-covid-19
# https://www.ecdc.europa.eu/en/publications-data/weekly-subnational-14-day-notification-rate-covid-19
# https://opendata.ecdc.europa.eu/covid19/subnationalcaseweekly/csv/
# https://opendata.ecdc.europa.eu/covid19/subnationalcaseweekly/json/
url = 'https://opendata.ecdc.europa.eu/covid19/subnationalcaseweekly/json/'
path = 'subnationalcaseweekly.json'
# Download only when there is no local copy, as in plot_choropleth.py. Before, a copy more than a day old was fetched again and written unchecked, which would replace ECDC's data of 21 January 2021, the source of europe.gif, with whatever the URL serves now. Delete the file to fetch it again.
if not os.path.isfile(path):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(path, 'wb') as f:
        f.write(r.content)
df_ecdc = pd.read_json(path)

# https://ec.europa.eu/eurostat/web/gisco/geodata/reference-data/administrative-units-statistical-units/nuts
url = 'https://gisco-services.ec.europa.eu/distribution/v2/nuts/download/ref-nuts-2021-01m.shp.zip'
if not os.path.isfile(os.path.basename(url)):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(os.path.basename(url), 'wb') as f:
        f.write(r.content)
    # Not named zip: that rebound the builtin, and the zip() over the Estonian codes below then failed.
    with zipfile.ZipFile(os.path.basename(url)) as archive:
        archive.extractall()
df_geo = pd.DataFrame()
for lvl in (1, 2, 3):
    path = f'NUTS_RG_01M_2021_4326_LEVL_{lvl}.shp'
    df_geo = pd.concat((df_geo, gpd.read_file(path)))

# print('\n'.join(sorted(set(df_geo['NAME_LATN'].unique()))))
# exit()
# print('\n'.join(sorted(set(df_geo['CNTR_CODE'].unique()))))
# print(df_geo[df_geo['CNTR_CODE'] == 'CH'][['NAME_LATN', 'NUTS_ID']])
# print(df_geo[df_geo['CNTR_CODE'] == 'IS'][['NAME_LATN', 'NUTS_ID']].to_string())
# exit()

# Scotland
# https://www.spatialdata.gov.scot/geonetwork/srv/api/records/f12c3826-4b4b-40e6-bf4f-77b9ed01dc14
# http://sedsh127.sedsh.gov.uk/Atom_data/ScotGov/ZippedShapefiles/SG_NHS_HealthBoards_2019.zip
path = 'SG_NHS_HealthBoards_2019.shp'
df_nhs_scotland = gpd.read_file(path).rename(columns={'HBCode': 'NUTS_ID'})
df_nhs_scotland['CNTR_CODE'] = 'UK'
# print(df_nhs_scotland.columns)
# print(df_nhs_scotland['HBName'])
# exit()

df_geo = pd.concat((
    df_geo,
    df_nhs_scotland,
    ))
# print(df_geo[['NUTS_ID', 'geometry', 'CNTR_CODE']])
# exit()

# Wales
# https://geoportal.statistics.gov.uk/datasets/87e71b2c79fc4ac894eeb79359cda131_0
path = 'Local_Health_Boards__December_2016__Boundaries.shp'
df_nhs_wales = gpd.read_file(path).rename(columns={'lhb16cd': 'NUTS_ID'})
df_nhs_wales['CNTR_CODE'] = 'UK'
# print(df_nhs_wales.columns)
# print(df_nhs_wales['lhb16cd'])
# print(df_nhs_wales['lhb16nm'])
# print(df_nhs_wales['lhb16nmw'])
# print(df_nhs_wales['bng_e'])
# exit()

df_geo = pd.concat((
    df_geo,
    df_nhs_wales,
    ))

df_ecdc.replace({'nuts_code': 'PTG301'}, 'PT18', inplace=True)  # Alentejo
df_ecdc.replace({'nuts_code': 'PTG302'}, 'PT15', inplace=True)  # Algarve
df_ecdc.replace({'nuts_code': 'PTG303'}, 'PT2', inplace=True)  # Região Autónoma dos Açores
df_ecdc.replace({'nuts_code': 'PTG304'}, 'PT16', inplace=True)  # Centro
df_ecdc.replace({'nuts_code': 'PTG305'}, 'PT17', inplace=True)  # Área Metropolitana de Lisboa
df_ecdc.replace({'nuts_code': 'PTG306'}, 'PT3', inplace=True)  # Região Autónoma da Madeira
df_ecdc.replace({'nuts_code': 'PTG307'}, 'PT11', inplace=True)

df_ecdc.replace({'nuts_code': 'BG412X'}, 'BG412', inplace=True)  # Sofia
df_ecdc.replace({'nuts_code': 'PL92X'}, 'PL92', inplace=True)  # Mazowiecki
df_ecdc.replace({'nuts_code': 'IS'}, 'IS0', inplace=True)  # Iceland

df_ecdc.replace({'nuts_code': 'NOG303'}, 'NO081', inplace=True)  # Oslo
df_ecdc.replace({'nuts_code': 'NOG311'}, 'NO0A1', inplace=True)  # Rogaland
df_ecdc.replace({'nuts_code': 'NOG315'}, 'NO0A3', inplace=True)  # Møre og Romsdal
df_ecdc.replace({'nuts_code': 'NOG318'}, 'NO071', inplace=True)  # Nordland
df_ecdc.replace({'nuts_code': 'NOG330'}, 'NO082', inplace=True)  # Viken
df_ecdc.replace({'nuts_code': 'NOG334'}, 'NO020', inplace=True)  # Innlandet
df_ecdc.replace({'nuts_code': 'NOG338'}, 'NO091', inplace=True)  # Vestfold og Telemark
df_ecdc.replace({'nuts_code': 'NOG342'}, 'NO092', inplace=True)  # Agder
df_ecdc.replace({'nuts_code': 'NOG346'}, 'NO0A2', inplace=True)  # Vestland
df_ecdc.replace({'nuts_code': 'NOG350'}, 'NO060', inplace=True)  # Trøndelag
df_ecdc.replace({'nuts_code': 'NOG354'}, 'NO074', inplace=True)  # Troms og Finnmark

df_ecdc.replace({'nuts_code': 'HR041'}, 'HR05', inplace=True)  # Grad Zagreb
df_ecdc.replace({'nuts_code': 'HR042'}, 'HR065', inplace=True)  # Zagrebačka županija
df_ecdc.replace({'nuts_code': 'HR043'}, 'HR064', inplace=True)  # Krapinsko-Zagorska Zupanija / Krapinsko-zagorska županija
df_ecdc.replace({'nuts_code': 'HR044'}, 'HR062', inplace=True)  # Varaždinska županija
df_ecdc.replace({'nuts_code': 'HR045'}, 'HR063', inplace=True)  # Koprivnicko-Krizevacka Zupanija / Koprivničko-križevačka županija
df_ecdc.replace({'nuts_code': 'HR046'}, 'HR061', inplace=True)  # Medimurska Zupanija / Međimurska županija
df_ecdc.replace({'nuts_code': 'HR047'}, 'HR021', inplace=True)  # Bjelovarsko-Bilogorska Zupanija / Bjelovarsko-bilogorska županija
df_ecdc.replace({'nuts_code': 'HR048'}, 'HR022', inplace=True)  # Viroviticko-Podravska Zupanija / Virovitičko-podravska županija
df_ecdc.replace({'nuts_code': 'HR049'}, 'HR023', inplace=True)  # Pozesko-Slavonska Zupanija / Požeško-slavonska županija
df_ecdc.replace({'nuts_code': 'HR04A'}, 'HR024', inplace=True)  # Brodsko-Posavska Zupanija / Brodsko-posavska županija
df_ecdc.replace({'nuts_code': 'HR04B'}, 'HR025', inplace=True)  # Osjecko-Baranjska Zupanija / Osječko-baranjska županija
df_ecdc.replace({'nuts_code': 'HR04C'}, 'HR026', inplace=True)  # Vukovarsko-Srijemska Zupanija / Vukovarsko-srijemska županija
df_ecdc.replace({'nuts_code': 'HR04D'}, 'HR027', inplace=True)  # Karlovacka Zupanija / Karlovačka županija
df_ecdc.replace({'nuts_code': 'HR04E'}, 'HR028', inplace=True)  # Sisacko-Moslavacka Zupanija / Sisačko-moslavačka županija

# Kirde-Eesti   EE00A  # Northeast
# Lõuna-Eesti   EE008  # South
# Kesk-Eesti   EE009  # Central
# Põhja-Eesti   EE001  # North
# Lääne-Eesti   EE004  # West
# https://en.wikipedia.org/wiki/NUTS_statistical_regions_of_Estonia
codes = [
    'EEG11212', 'EE001',  # Harju Maakond
    'EEG11213', 'EE004',  # Hiiu Maakond
    'EEG11214', 'EE00A',  # Ida-Viru Maakond
    'EEG11215', 'EE009',  # Järva Maakond
    'EEG11216', 'EE008',  # Jõgeva Maakond
    'EEG11217', 'EE009',  # Lääne-Viru Maakond
    'EEG11218', 'EE004',  # Lääne Maakond
    'EEG11219', 'EE004',  # Pärnu Maakond
    'EEG11220', 'EE008',  # Põlva Maakond
    'EEG11221', 'EE009',  # Rapla Maakond
    'EEG11222', 'EE004',  # Saare Maakond
    'EEG11223', 'EE008',  # Tartu Maakond
    'EEG11224', 'EE008',  # Valga Maakond
    'EEG11225', 'EE008',  # Viljandi Maakond
    'EEG11226', 'EE008',  # Võru Maakond
    ]
it = iter(codes)
for code1, code2 in zip(it, it):
    df_ecdc.replace({'nuts_code': code1}, code2, inplace=True)

df_ecdc = df_ecdc.loc[df_ecdc['nuts_code'] != 'FRY3']  # Guyane
df_ecdc = df_ecdc.loc[df_ecdc['nuts_code'] != 'FRY4']  # Reunion
df_ecdc = df_ecdc.loc[df_ecdc['nuts_code'] != 'FRY5']  # Mayotte

# GL = Greenland
# IM = Isle of Man

x = set(df_ecdc['nuts_code'].unique())
y = df_geo['NUTS_ID']
print('diff_ecdc_not_geo', sorted(set(x) - set(y)))

vmax = 600  # Maximum in the spring. La Rioja, Spain, W14.
vmax = 1000
paths = []

# Restrict to vectors in ECDC set so they can all be set to zero subsequently.
df_geo = GeoDataFrame(pd.merge(
    df_ecdc['nuts_code'].drop_duplicates(),
    df_geo[['NUTS_ID', 'geometry']],
    left_on=['nuts_code'],
    right_on=['NUTS_ID'],
    how='left',
    ))

for cmap in (
        # 'inferno',
        'viridis',
        # 'plasma',
        # 'magma',
        ):
    for year_week in sorted(df_ecdc['year_week'].unique()):
        # df =
        # x = df_ecdc['nuts_code']
        # exit()

        # print(len(df_merged))
        # print()
        # print(df_merged.columns)
        # print(df_merged['NUTS_ID'])
        # print(df_merged[df_merged['NUTS_ID'] == 'FI200'])
        # exit()

        # scheme = mapclassify.Quantiles(gpd_per_person, k=5)

        # geoplot.choropleth(
        #     df_merged, hue='rate_14_day_per_100k',
        #     scheme=scheme,
        #     cmap='Greens', figsize=(8, 4)
        # )

        fig, ax = plt.subplots()
        fig.set_size_inches(16/2, 9/2)

        for df in (df_geo, df_nhs_scotland, df_nhs_wales):
            df_week = df_ecdc.loc[df_ecdc['year_week'] == year_week]
            # Sum cases across groups with same NUTS code.
            df_week = pd.DataFrame(df_week.groupby('nuts_code').sum())
            df_merged = GeoDataFrame(pd.merge(
                df_week,
                df,
                left_on=['nuts_code'],
                right_on=['NUTS_ID'],
                how='right',
                # df['rate_14_day_per_100k'])
                ))  # .fillna(value={'rate_14_day_per_100k': 0})).dropna(subset=[''])
            df_merged['rate_14_day_per_100k'] = df_merged['rate_14_day_per_100k'].fillna(0)

            # print(df['NUTS_ID'].unique())
            # print(df_ecdc[df_ecdc['year_week'] == year_week]['nuts_code'].unique())
            # print(df_ecdc['nuts_code'].unique())
            # print(df_merged['NUTS_ID'].unique())
            # print(df_merged.columns)

            # print(df_merged.columns)
            # print(df_merged[['geometry', 'NUTS_ID', 'rate_14_day_per_100k']])
            # pd.set_option("max_rows", None)
            # print(df_merged[['geometry', 'NUTS_ID']])

            m = df_merged.plot(
                ax=ax,
                column='rate_14_day_per_100k',
                cmap=cmap,
                # linewidth=0.1,
                # edgecolor='black',
                legend=True,
                legend_kwds={
                    'label': 'ECDC weekly subnational 14-day notification rate\nof new COVID-19 cases per 100,000 inhabitants',
                    'orientation': "vertical",
                    'norm': Normalize(vmin=0, vmax=vmax),
                    # 'properties': {
                    #     'size': 'xx-small',
                    #     'width': '50%',
                    #     },
                    },
                # missing_kwds={
                #     'rate_14_day_per_100k': 0,
                #     # 'color': 'white',
                #     },
                vmin=0,
                vmax=vmax,
                )
            break

        # Remove frame.
        ax.axis('off')

        # ax.colorbar(m[0])

        # adjust plot domain to focus on EU region
        plt.xlim(-25, 35)
        plt.ylim(33, 72)

        path = f'europe_{year_week}.png'
        ax.set_title(year_week)
        plt.savefig(path, dpi=150)
        paths.append(path)

        print(path)

    path_gif = 'europe.gif'
    # Do custom frame lengths with imagemagick.
    command = 'convert -delay 25 {} -delay 400 {} {}'.format(
        ' '.join(paths[:-1]), paths[-1], path_gif)
    print(command)
    os.system(command)

    command = 'ffmpeg -framerate 1/0.25 -pattern_type glob -i europe_*.png -pix_fmt yuv420p -tune stillimage -preset veryslow -crf 0 europe.mp4'
    command = 'ffmpeg -i europe.gif -movflags faststart -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" europe.mp4'
    print(command)
    os.system(command)

    for path in paths:
        continue
        os.remove(path)

# ffmpeg -i denmark.gif -movflags faststart -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" denmark.mp4
