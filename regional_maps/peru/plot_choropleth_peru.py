import os

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from geopandas import GeoDataFrame
from matplotlib.colors import Normalize

# https://www.datosabiertos.gob.pe/dataset/casos-positivos-por-covid-19-ministerio-de-salud-minsa
path = 'positivos_covid.csv'
df_cases = pd.read_csv(
    path, sep=';', parse_dates=['FECHA_RESULTADO'])
print(df_cases['DEPARTAMENTO'].unique())
print(df_cases.columns)
# df_cases[df_cases['DEPARTAMENTO'] == 'LIMA REGION'] = 'LIMA'

# https://en.wikipedia.org/wiki/Regions_of_Peru#Departments
url = 'https://en.wikipedia.org/wiki/Regions_of_Peru#Departments'
df_pop = pd.read_html(url)[1]
df_pop['ISO'] = df_pop['ISO'].str[3:]
df_pop['Region'] = df_pop['Region'].str.upper()
df_pop.replace('HUÁNUCO', 'HUANUCO', inplace=True)
df_pop.replace('SAN MARTÍN', 'SAN MARTIN', inplace=True)
df_pop.replace('APURÍMAC', 'APURIMAC', inplace=True)
df_pop.replace('JUNÍN', 'JUNIN', inplace=True)
df_pop.replace('LIMA', 'LIMA REGION', inplace=True)
# df_pop[df_pop['Region'] == 'LIMA']['Population'] = 9485405
df_pop = pd.concat((df_pop, pd.DataFrame([{'Region': 'LIMA', 'Population': 8574974}])), ignore_index=True)
print(df_pop)
print(set(df_pop['Region']) - set(df_cases['DEPARTAMENTO']))
print(set(df_cases['DEPARTAMENTO']) - set(df_pop['Region']))

df_cases = df_cases.pivot_table(
    # df_cases,
    values='UUID',
    columns='DEPARTAMENTO',
    index=['FECHA_RESULTADO'],
    aggfunc='count',
    ).fillna(0)  # .reset_index()
# df_cases.set_index('FECHA_RESULTADO', inplace=True)
print(df_cases)

# df_cases_daily = df_cases
df_cases = pd.DataFrame(df_cases.rolling(7).sum())

# # https://en.wikipedia.org/wiki/Template:COVID-19_pandemic_data/Peru_medical_cases#By_departments
# url = 'https://en.wikipedia.org/wiki/Template:COVID-19_pandemic_data/Peru_medical_cases'
# df_cases = pd.read_html(url)[1].rename(columns={
#     'Copenhagen': 'København',
#     }).fillna(0)
# print(df_cases)

# https://data.humdata.org/dataset/limites-de-peru
path = 'per_admbnda_adm0_ign_20200714.shp'
path = 'per_admbnda_adm1_ign_20200714.shp'
df_geo = gpd.read_file(path)
df_geo['ADM1_ES'] = df_geo['ADM1_ES'].str.upper()
print(df_geo.columns)
print(df_geo['ADM1_ES'])

vmax = 200
maxi = 0

paths = []
# Skip first (rolling sum wrong) and last (case count wrong) days
for date in sorted(df_cases.index.values)[7:]:
    df_merged = pd.merge(
        df_cases.loc[df_cases.index == date].transpose(),
        df_pop,
        left_on=['DEPARTAMENTO'],
        right_on=['Region'],
        # how='outer',
        )
    df_merged.rename(columns={
            df_merged.columns.tolist()[0]: 'cases',
            }, inplace=True)
    df_merged['CasesPer100k'] = 100000 * df_merged['cases'] / df_merged['Population']
    print('max', df_merged['CasesPer100k'].max())
    maxi = max(maxi, df_merged['CasesPer100k'].max())
    df_merged = GeoDataFrame(pd.merge(
        df_merged,
        df_geo,
        left_on=['Region'],
        right_on=['ADM1_ES'],
        # how='right',
        how='inner',
        ))
    # df_merged = df_merged.astype({'cases': 'int32'})

    fig, ax = plt.subplots()
    fig.set_size_inches(16 / 2, 9 / 2)

    df_merged.plot(
        ax=ax,
        column='CasesPer100k',
        cmap='viridis',
        # linewidth=0.1,
        # edgecolor='black',
        legend=True,
        legend_kwds={
            'label': 'Casos semanales de COVID-19 por 100k',
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

    ax.axis('off')

    # plt.xlim(6, 16)
    # plt.ylim(54, 58)

    # a = plt.axes([.65, .6, .2, .2], facecolor='w')
    # plt.title('Cases')
    # df_cases_daily[df_cases_daily['date'] >= date].plot.bar(x='date', y='cases')

    path = f'peru_{str(date)[:10]}.png'
    ax.set_title(str(date)[:10])
    plt.savefig(path, dpi=150)
    paths.append(path)
    print(path)

print('MAXIMUM', maxi)

path_gif = 'peru.gif'
# Do custom frame lengths with imagemagick.
command = 'convert -delay 25 {} -delay 400 {} {}'.format(
    ' '.join(paths[:-1]), paths[-1], path_gif)
print(command)
os.system(command)

command = 'ffmpeg -i peru.gif -movflags faststart -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" peru.mp4'
os.system(command)
