import os

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from geopandas import GeoDataFrame
from matplotlib.colors import Normalize

# https://covid19.ssi.dk/overvagningsdata/download-fil-med-overvaagningdata
path = 'Municipality_cases_time_series.csv'
df_ssi = pd.read_csv(path, sep=';').rename(columns={
    'Copenhagen': 'København',
    # 'Høje-Taastrup': 'Høje Taastrup',
    })
print(df_ssi)
df_ssi.set_index('date_sample', inplace=True)
print(df_ssi)
df_ssi = pd.DataFrame(df_ssi.rolling(7).sum())
print(df_ssi)

# https://github.com/Neogeografen/dagi/blob/master/geojson/kommuner.geojson
path = 'kommuner.geojson'
df_kommuner = gpd.read_file(path)
df_kommuner['KOMNAVN'] = df_kommuner['KOMNAVN'].replace(
    'Høje Taastrup', 'Høje-Taastrup')

# https://www.statistikbanken.dk/statbank5a/selectvarval/define.asp?PLanguage=0&subword=tabsel&MainTable=FOLK1A
path = '202012249554308885851FOLK1A35728322320.csv'
path = '2020122410143308885851FOLK1A36126431121.csv'
df_dst = pd.read_csv(path, sep=';', encoding="ISO-8859-1")
print(df_dst)

print(set(df_ssi.columns) - set(df_kommuner['KOMNAVN'].values))

cmap = 'viridis'
vmax = 500

paths = []
# Skip first (rolling sum wrong) and last (case count wrong) days
for date in sorted(df_ssi.index.values)[7 - 1:-4]:
    df_merged = pd.merge(
        df_ssi.loc[df_ssi.index == date].transpose().reset_index(),
        df_dst,
        left_on=['index'],
        right_on=['Kommune'],
        how='outer',
        )
    df_merged.rename(columns={
            df_merged.columns.tolist()[1]: 'cases',
            }, inplace=True)
    df_merged['CasesPer100k'] = 100000 * df_merged['cases'] / df_merged['Personer']
    print(df_merged['CasesPer100k'].max())
    df_merged = GeoDataFrame(pd.merge(
        df_merged,
        df_kommuner,
        left_on=['index'],
        right_on=['KOMNAVN'],
        # how='right',
        how='inner',
        ))
    df_merged = df_merged.astype({'cases': 'int32'})

    fig, ax = plt.subplots()
    fig.set_size_inches(16/2, 9/2)

    df_merged.plot(
        ax=ax,
        column='CasesPer100k',
        cmap=cmap,
        # linewidth=0.1,
        # edgecolor='black',
        legend=True,
        legend_kwds={
            'label': 'Weekly COVID-19 cases per 100k',
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

    plt.xlim(6, 16)
    plt.ylim(54, 58)

    path = f'denmark_{date}.png'
    ax.set_title(date)
    plt.savefig(path, dpi=150)
    paths.append(path)
    print(path)

path_gif = 'denmark.gif'
# Do custom frame lengths with imagemagick.
command = 'convert -delay 25 {} -delay 400 {} {}'.format(
    ' '.join(paths[:-1]), paths[-1], path_gif)
print(command)
os.system(command)

command = 'ffmpeg -i denmark.gif -movflags faststart -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" denmark.mp4'
os.system(command)
