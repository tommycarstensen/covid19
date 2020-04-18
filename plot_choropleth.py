# Tommy Carstensen, March 2020

import argparse
from datetime import datetime
import pandas as pd
import geopandas as gpd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import imageio
import shutil
import requests
import os


def main():

    args = parse_args()

    df1, df2 = parse_data(args)

    df2['dateRep'] = pd.to_datetime(df2['dateRep'], format='%d/%m/%Y')
    df0 = df2[['alpha3', 'dateRep', 'cases', 'deaths']].groupby(['alpha3', 'dateRep']).sum()

    # print(df2[df2['alpha3'] == 'ABW'])
    # ABW 16 cases
       # dateRep  day  month  year  cases  deaths countriesAndTerritories alpha2 alpha3  popData2018
# 489 2020-04-16   16      4  2020      1       1                   Aruba     AW    ABW     105845.0
# 490 2020-04-15   15      4  2020      0       0                   Aruba     AW    ABW     105845.0
# 491 2020-04-14   14      4  2020      0       0                   Aruba     AW    ABW     105845.0
# 492 2020-04-13   13      4  2020      0       0                   Aruba     AW    ABW     105845.0
# 493 2020-04-12   12      4  2020      6       0                   Aruba     AW    ABW     105845.0
# 494 2020-04-11   11      4  2020      4       0                   Aruba     AW    ABW     105845.0
# 495 2020-04-10   10      4  2020      5       0                   Aruba     AW    ABW     105845.0
    # print(df2[df2['alpha3'] == 'ZWE'])
#          dateRep  day  month  year  cases  deaths countriesAndTerritories alpha2 alpha3  popData2018
# 11125 2020-04-16   16      4  2020      6       0                Zimbabwe     ZW    ZWE   14439018.0
# 11126 2020-04-15   15      4  2020      0       0                Zimbabwe     ZW    ZWE   14439018.0
# 11127 2020-04-14   14      4  2020      3       0                Zimbabwe     ZW    ZWE   14439018.0
# 11128 2020-04-13   13      4  2020      0       0                Zimbabwe     ZW    ZWE   14439018.0
# 11129 2020-04-12   12      4  2020      3       0                Zimbabwe     ZW    ZWE   14439018.0
# 11130 2020-04-11   11      4  2020      0       0                Zimbabwe     ZW    ZWE   14439018.0
# 11131 2020-04-10   10      4  2020      0       1                Zimbabwe     ZW    ZWE   14439018.0
# 11132 2020-04-09    9      4  2020      1       1                Zimbabwe     ZW    ZWE   14439018.0

    cmap = 'OrRd'
    cmaps = {
        'cumulated': {'cases': 'OrRd', 'deaths': 'PuRd'},
        'per week': {'cases': 'YlGnBu', 'deaths': 'PuBuGn'},
        }
    for period in ('per week', 'cumulated'):  # cumulated or weekly
    # for period in ('', 'per week', ''):  # cumulated or weekly

        # Do groupby and sum for UK, Denmark (Denmark, Greenland, Faroe Islands) and others? I can't remember.
        if period == 'per week':  # rolling sum 7 days
            df2 = df0.reset_index(level='alpha3')
            df2 = df2.groupby('alpha3')
            df2 = df2.rolling(window=7).sum()
            df2 = df2.reset_index()
            # Fill intermittently missing data for each alpha3 group after resampling grouped dataframe.
            df2 = df2.set_index('dateRep').groupby('alpha3').resample('1D').ffill().reset_index(level='dateRep').reset_index(drop=True)
        else:  # cumulative
            # Cumulated sum by country (alpha3).
            df2 = df0.groupby(level='alpha3').cumsum().reset_index()
            # Fill intermittently missing data for each alpha3 group after resampling grouped dataframe.
            df2 = df2.set_index('dateRep').groupby('alpha3').resample('1D').ffill().reset_index(level='dateRep').reset_index(drop=True)

        df1.rename(columns={'iso_a3': 'alpha3'}, inplace=True)
        # df = pd.merge(df1, df2, on=['alpha3'], how='left').fillna(value={'cases': 0, 'deaths': 0})

        # https://matplotlib.org/examples/color/colormaps_reference.html
        # maxDateRep = max(df2['dateRep'].unique())
        # for boolLog in (False, True,):
        boolLog = True
        for column in ('cases', 'deaths'):
            cmap = cmaps[period][column]
            # df = pd.merge(df1, df2[df2['dateRep'] == maxDateRep], on=['alpha3'], how='left').fillna(value={'cases': 0, 'deaths': 0})
            # zlim_max = max(10**6 * df[column] / df['pop_est'])
            valuemax = {
            'cases': 10000,  # cum Iceland 5262.5; week Luxembourg 3.219995
            'deaths': 1000,  # Spain 399.5; week Belgium 2.282052
            }[column]
                # ax.clim(0, 100)
            # for cmap in ('OrRd', 'YlGn'):
            # import matplotlib as mpl
            # cb = ColorbarBase(
            #     ax, cmap=mpl.cm.cool,
            #     norm = Normalize(vmin=0, vmax=vmax),
            #     orientation = 'horizontal',
            #     )
            images = []
            paths = []
            dateToday = max(df2['dateRep'])
            for DateRep in sorted(df2['dateRep'].unique()):
                if dateToday.weekday() != pd.to_datetime(DateRep).weekday():
                    continue
                try:
                    dateString = pd.to_datetime(DateRep).strftime('%Y-%m-%d')
                except ValueError:
                    continue
                print(cmap, column, dateString)
                df = pd.merge(df1, df2[df2['dateRep'] == DateRep], on=['alpha3'], how='left').fillna(value={'cases': 0, 'deaths': 0})
                if boolLog is True:
                    df['proportion'] = np.log10(10**6 * df[column] / df['pop_est'])
                    df.loc[df['proportion'] == -math.inf, 'proportion'] = None
                    vmax = int(math.log10(valuemax))
                    vmin = {
                        'cumulated': {'cases': -2, 'deaths': -3},
                        'per week': {'cases': -2, 'deaths': -3},  # cases India -3.107866, deaths China -2.83863
                        }[period][column]
                    label = 'log10 of {} per 1 million'.format(column)
                else:
                    df['proportion'] = 10**6 * df[column] / df['pop_est']
                    vmin = 0
                    vmax = 10 ** math.ceil(math.log10(max(0.001, max(df['proportion']))))
                    label = '{} per 1 million'.format(column)
                # df = df[df['proportion'] != -math.inf]
                # df_plot['proportion'].multiply(10**6)  # per million
                # https://geopandas.org/mapping.html
                fig, ax = plt.subplots(1, 1)
                ax.axis('off')
                ax.set_title(
                    'CoViD19 {} {}\n{}'.format(
                        column, period, dateString), fontsize='large')
                # vmax = int(math.log10(max(0.001, max(df['proportion']))))
                df.plot(
                    column='proportion',
                    cmap=cmap,
                    linewidth=0.1,
                    ax=ax,
                    edgecolor='black',
                    legend=True,
                    legend_kwds = {
                        'label': label,
                        'orientation': "horizontal",
                        'norm': Normalize(vmin=0, vmax=vmax),
                        # 'properties': {'size': 'xx-small'},
                        },
                    missing_kwds={'color': 'white'},
                    vmin = vmin,
                    # vmax = vmax,
                    vmax = vmax,
                    )

                # https://stackoverflow.com/questions/53158096/editing-colorbar-legend-in-geopandas
                # pcm = ax[0].pcolor(X, Y, Z,
                   # norm=colors.LogNorm(vmin=Z.min(), vmax=Z.max()),
                   # cmap='PuBu_r')
                # fig.colorbar(pcm, ax=ax[0], extend='max')

                # plt.tight_layout()
                path = 'covid19_{}{}_{}_log{}_{}.png'.format(
                    column, period.replace(' ',''), cmap, boolLog, dateString,
                    )
                # fig.colorbar()
                plt.savefig(path, dpi=100)
                images.append(imageio.imread(path))
                paths.append(path)
                plt.close()
                print(path)

            shutil.copyfile(path, 'covid19_{}{}_{}_log{}.png'.format(
                    column, period.replace(' ',''), cmap, boolLog))
            path_gif = 'covid19_{}{}_{}_log{}.gif'.format(
                column, period.replace(' ', ''), cmap, boolLog)
            # Do custom frame lengths with imagemagick.
            command = 'convert -delay 50 {} -delay 300 {} {}'.format(
                ' '.join(paths[:-1]), paths[-1], path_gif)
            os.system(command)
            for path in paths:
                if '2020-02-21' in path:
                    continue
                os.remove(path)

    return


def parse_data(args):

    url = 'https://opendata.ecdc.europa.eu/covid19/casedistribution/csv'
    r = requests.get(url)
    with open('csv', 'w') as f:
        f.write(r.text)

    df_covid19 = pd.read_csv('csv')

    df_covid19.rename(columns={
        'geoId': 'alpha2',
        'countryterritoryCode': 'alpha3',
        }, inplace=True)

    # # Get ISO 3166-1 alpha-2 and alpha-3 codes for each country.
    # df_iso3166 = pd.DataFrame(iso3166.countries)

    # # Merge geo data with alpha2 codes on alpha3 codes.
    # df1 = pd.merge(df_geo, df_iso3166, on=['alpha3'])

    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world = world[(world.pop_est>0) & (world.name!="Antarctica")]
    df1 = world
    # Fix error in dataset.
    # https://github.com/geopandas/geopandas/issues/1041
    print(df1[df1['iso_a3'] == '-99']['name'])
    df1.loc[df1['name'] == 'France', 'iso_a3'] = 'FRA'
    df1.loc[df1['name'] == 'Norway', 'iso_a3'] = 'NOR'
    df1.loc[df1['name'] == 'Somaliland', 'iso_a3'] = 'SOM'
    df1.loc[df1['name'] == 'Kosovo', 'iso_a3'] = 'RKS'
    print(df1[df1['iso_a3'] == '-99']['name'])

    # print(set(df_covid19['alpha2'].values) - set(df_iso3166['alpha2'].values))
    # df_covid19.loc[df_covid19['alpha2'] == 'UK', 'alpha2'] = 'GB'

    print(set(df_covid19['alpha3'].values) - set(df1['iso_a3'].values))
    print(set(df1['iso_a3'].values) - set(df_covid19['alpha3'].values))
    for x in set(df_covid19['alpha3'].values) - set(df1['iso_a3'].values):
        print(df_covid19[df_covid19['alpha3'] == x]['countriesAndTerritories'].unique())

    # # Merge covid19 data with alpha3 codes on alpha2 codes.
    # df2 = pd.merge(df_covid19, df_iso3166, on=['alpha2'])

    df2 = df_covid19

    # # Merge geo data with covid19 data on alpha2 codes.
    # df = pd.merge(df, df_covid19.rename(columns={'GeoId': 'alpha2'}), on=['alpha2'])[['admin', 'alpha3', 'cases', 'deaths', 'dateRep']]

    return df1, df2


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--date', default=datetime.today().strftime('%Y-%m-%d'),
        help='Date in ISO 8601 format YYYY-MM-DD',
        required=False,
        )

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()