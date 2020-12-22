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
import time


def main():

    args = parse_args()

    df1, df2, df_owid = parse_data(args)

    df2['dateRep'] = pd.to_datetime(df2['dateRep'], format='%d/%m/%Y')
    df0 = df2[[
        'alpha3', 'dateRep', 'cases_weekly', 'deaths_weekly']].groupby([
            'alpha3', 'dateRep']).sum()

    df1.rename(columns={'iso_a3': 'alpha3'}, inplace=True)

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

    plot_owid(df_owid, df1)

    cmap = 'OrRd'
    cmaps = {
        'cumulated': {'cases': 'OrRd', 'deaths': 'PuRd'},
        'per week': {'cases': 'YlGnBu', 'deaths': 'PuBuGn'},
        }
    for period in ('per week', 'cumulated'):  # cumulated or weekly
        print(period)
    # for period in ('', 'per week', ''):  # cumulated or weekly

        # Do groupby and sum for UK, Denmark (Denmark, Greenland, Faroe Islands) and others? I can't remember.
        if period == 'per week':  # rolling sum 7 days
            df2 = df0.reset_index(level='alpha3')
            df2 = df2.groupby('alpha3')
            df2 = df2.rolling(window=7).sum()
            df2 = df2.reset_index()
            # Fill intermittently missing data for each alpha3 group after resampling grouped dataframe.
            df2 = df2.set_index('dateRep').groupby('alpha3').resample('1D').ffill().reset_index(level='dateRep').reset_index(drop=True)
            # df2 = df2.set_index('dateRep').groupby('alpha3').resample('1W').ffill().reset_index(level='dateRep').reset_index(drop=True)
        else:  # cumulative
            # Cumulated sum by country (alpha3).
            df2 = df0.groupby(level='alpha3').cumsum().reset_index()
            # Fill intermittently missing data for each alpha3 group after resampling grouped dataframe.
            df2 = df2.set_index('dateRep').groupby('alpha3').resample('1D').ffill().reset_index(level='dateRep').reset_index(drop=True)
            # df2 = df2.set_index('dateRep').groupby('alpha3').resample('1W').ffill().reset_index(level='dateRep').reset_index(drop=True)

        # df = pd.merge(df1, df2, on=['alpha3'], how='left').fillna(value={'cases': 0, 'deaths': 0})

        # https://matplotlib.org/examples/color/colormaps_reference.html
        # maxDateRep = max(df2['dateRep'].unique())
        # for boolLog in (False, True,):
        boolLog = True
        for key in ('cases', 'deaths'):
            column = key + '_weekly'
            print(column)
            cmap = cmaps[period][key]
            # df = pd.merge(df1, df2[df2['dateRep'] == maxDateRep], on=['alpha3'], how='left').fillna(value={'cases': 0, 'deaths': 0})
            # zlim_max = max(10**6 * df[column] / df['pop_est'])
            # 
            valuemax = {
                'cases': 10000,  # cum Iceland 5262.5; week Luxembourg 3.219995
                'deaths': 1000,  # Spain 539.7; week Belgium 2.282052
                }[key]
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
                # if (dateToday - DateRep).days % 14 != 0:
                #     continue
                if dateToday.weekday() != pd.to_datetime(DateRep).weekday():
                    continue
                try:
                    dateString = pd.to_datetime(DateRep).strftime('%Y-%m-%d')
                except ValueError:
                    continue
                print(cmap, column, dateString)
                df = pd.merge(
                    df1, df2[df2['dateRep'] == DateRep],
                    on=['alpha3'], how='left'
                    ).fillna(value={'cases': 0, 'deaths': 0})
                if boolLog is True:
                    df['proportion'] = np.log10(
                        10**6 * df[column] / df['pop_est'])
                    df.loc[df['proportion'] == -math.inf, 'proportion'] = None
                    vmax = int(math.log10(valuemax))
                    vmin = {
                        'cumulated': {'cases': -2, 'deaths': -3},
                        'per week': {'cases': -2, 'deaths': -3},  # cases India -3.107866, deaths China -2.83863
                        }[period][key]
                    label = 'log10 of {} per 1 million'.format(key)
                else:
                    df['proportion'] = 10**6 * df[column] / df['pop_est']
                    vmin = 0
                    vmax = 10 ** math.ceil(math.log10(max(
                        0.001, max(df['proportion']))))
                    label = '{} per 1 million'.format(key)
                # df = df[df['proportion'] != -math.inf]
                # df_plot['proportion'].multiply(10**6)  # per million
                # https://geopandas.org/mapping.html
                fig, ax = plt.subplots(1, 1)
                ax.axis('off')
                ax.set_title(
                    'CoViD19 {} {}\n{}'.format(
                        key, period, dateString), fontsize='large')
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
                    key, period.replace(' ',''), cmap, boolLog, dateString,
                    )
                # fig.colorbar()
                plt.savefig(path, dpi=80)
                images.append(imageio.imread(path))
                paths.append(path)
                plt.close()
                print(path)

            shutil.copyfile(path, 'covid19_{}{}_{}_log{}.png'.format(
                    key, period.replace(' ',''), cmap, boolLog))
            path_gif = 'covid19_{}{}_{}_log{}.gif'.format(
                key, period.replace(' ', ''), cmap, boolLog)
            # Do custom frame lengths with imagemagick.
            command = 'convert -delay 50 {} -delay 400 {} {}'.format(
                ' '.join(paths[:-1]), paths[-1], path_gif)
            os.system(command)
            for path in paths:
                if '2020-02-21' in path:
                    continue
                os.remove(path)

    return


def plot_owid(df, df_geo):

    print(df[['tests_units', 'location']].drop_duplicates().dropna().to_string())
    print(df['tests_units'].unique())

    # Rename column to allow merger.
    df = df.rename(columns={'iso_code': 'alpha3'})

    # Filter only columns of interest.
    df = df[[
        'total_tests_per_thousand',
        'new_tests_per_thousand',
        'alpha3', 'date',
        ]]

    # Get latest date for which various countries all have data.
    dateMax = min((
        max(df[(df['alpha3'] == alpha3) & (df['total_tests_per_thousand'] > 0)]['date'].unique()) for alpha3 in (
            'USA', 'DEU', 'KOR', 'JPN', 'ISL',)))
    dateMax = datetime.strptime(dateMax, '%Y-%m-%d')

    # Convert date string to datetime object for doing resampling.
    df['date'] = pd.to_datetime(df['date'])

    # Fill missing dates for each alpha3 group by resampling and interpolation.
    # Fill missing data *before* converting cumulative sums to daily values.
    # Resample and interpolate.
    # resample: For example Qater missing entirely 3rd of March.
    # interpolate: For example Australia value missing 2nd of March, but not 1st and 3rd.
    df = df.set_index('date').groupby('alpha3').resample('1D').ffill()
    df = df.reset_index(level='alpha3', drop=True)
    # https://stackoverflow.com/a/48027252/778533
    print('interpolating')
    df['total_tests_per_thousand'] = df[['total_tests_per_thousand', 'alpha3']].groupby('alpha3').transform(pd.DataFrame.interpolate)
    print('interpolated')

    # Convert cumulative sum to daily values *after* filling missing values.
    # For example Australia blank on 2nd of March.
    # cat owid.csv | cut -d"," -f1,3,14,15 | grep AUS
    # AUS,2020-03-01,0.115,
    # AUS,2020-03-02,,
    # AUS,2020-03-03,0.12,
    # https://stackoverflow.com/a/36452075/778533
    df['daily_tests_per_thousand'] = df['total_tests_per_thousand'].diff().fillna(df['new_tests_per_thousand'])

    # Calculate weekly values from calculated daily values.
    # Do a 7 day rolling average.
    df['weekly_tests_per_thousand'] = df['daily_tests_per_thousand'].rolling(window=7).sum()

    # Reset the index prior to plotting.
    # df_owid = df.reset_index(level='alpha3', drop=True).reset_index()
    df_owid = df.reset_index()

    d_paths = {
        'weekly_tests_per_thousand': [],
        'total_tests_per_thousand': [],
        }
    for date in sorted(df_owid['date'].unique()):
        print(date)
        if dateMax.weekday() != pd.to_datetime(date).weekday():
            continue
        dateString = pd.to_datetime(date).strftime('%Y-%m-%d')

        for column, cmap, vmin, vmax in (
            # vmin -3 because only three significant digits in csv file
            # vmax weekly: ISL,2020-04-05,7.243
            # vmax total: ISL,2020-04-20,127.58
            ('weekly_tests_per_thousand', 'Blues', -3, 1),
            ('total_tests_per_thousand', 'Greens', -3, 3),
            ):
            df_merged = pd.merge(
                df_geo,
                df_owid[df_owid['date'] == date],
                # df_owid,
                on=['alpha3'],
                how='left',
                ).fillna(
                value={
                'total_tests_per_thousand': 0,
                'weekly_tests_per_thousand': 0,
                })

            # df = df_merged[df_merged['date'] == date]
            df = df_merged

            df['column'] = np.log10(df[column])
            df.loc[df['column'] == -math.inf, 'column'] = None
            fig, ax = plt.subplots(1, 1)
            ax.axis('off')
            ax.set_title(
                'CoViD19 {}\n{}'.format(
                    column.replace('_', ' '), dateString),
                fontsize='large',
                )
            label = 'log10 of {}'.format(column.replace('_', ' '))
            df.plot(
                column='column',
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
                missing_kwds={'color': 'grey'},
                vmin = vmin,
                # vmax = vmax,
                vmax = vmax,
                )
            path = 'covid19_{}_{}.png'.format(
                column, dateString,
                )
            plt.savefig(path, dpi=80)
            d_paths[column].append(path)
            plt.close()

    for column, paths in d_paths.items():
        path_gif = 'covid19_{}.gif'.format(column)
        # Do custom frame lengths with imagemagick.
        command = 'convert -delay 50 {} -delay 400 {} {}'.format(
            ' '.join(paths[:-1]), paths[-1], path_gif)
        os.system(command)
        for path in paths:
            os.remove(path)

    return


def download_and_read(url, path, func):

    # df_owid = pd.read_html(url)

    if os.path.isfile(path) and time.time() - os.path.getmtime(path) < 2 * 3600:
        pass
    else:
        print(url)
        r = requests.get(url)
        with open(path, 'w') as f:
            f.write(r.text)

    df = func(path)

    return df


def parse_data(args):

    url = 'https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv'
    path = 'owid.csv'
    df_owid = download_and_read(url, path, pd.read_csv)

    url = 'https://github.com/OxCGRT/covid-policy-tracker/raw/master/data/OxCGRT_latest.csv'
    path = 'bsg.csv'
    df_bsg = download_and_read(url, path, pd.read_csv)

    url = 'https://opendata.ecdc.europa.eu/covid19/casedistribution/csv'
    path = 'ecdc.csv'
    # url = 'https://opendata.ecdc.europa.eu/covid19/casedistribution/json'
    # path = 'ecdc.json'
    df_ecdc = download_and_read(url, path, pd.read_csv)

    df_ecdc.rename(columns={
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

    print(set(df_ecdc['alpha3'].values) - set(df1['iso_a3'].values))
    print(set(df1['iso_a3'].values) - set(df_ecdc['alpha3'].values))
    for x in set(df_ecdc['alpha3'].values) - set(df1['iso_a3'].values):
        print(df_ecdc[df_ecdc['alpha3'] == x]['countriesAndTerritories'].unique())

    # # Merge covid19 data with alpha3 codes on alpha2 codes.
    # df2 = pd.merge(df_covid19, df_iso3166, on=['alpha2'])

    # # Merge geo data with covid19 data on alpha2 codes.
    # df = pd.merge(df, df_covid19.rename(columns={'GeoId': 'alpha2'}), on=['alpha2'])[['admin', 'alpha3', 'cases', 'deaths', 'dateRep']]

    return df1, df_ecdc, df_owid


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
