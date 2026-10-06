from bokeh.io import (
    save,
    show,
    output_file,
    output_notebook,
    reset_output,
    export_png,
    )
from bokeh.plotting import figure
from bokeh.models import (
    GeoJSONDataSource, ColumnDataSource, ColorBar, Slider, Spacer,
    HoverTool, TapTool, Panel, Tabs, Legend, Toggle, LegendItem,
)
from bokeh.models import GeoJSONDataSource, LinearColorMapper, ColorBar

from bokeh.palettes import brewer
from bokeh.models.callbacks import CustomJS
from bokeh.models.widgets import Div
from bokeh.layouts import widgetbox, row, column
from matplotlib import pyplot as plt
from matplotlib.colors import rgb2hex
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase

import zipfile
import os
import requests
from datetime import datetime
import argparse
import json

import geopandas as gpd
import pandas as pd
import bokeh
import math

import iso3166

# import our map function
from interactive_maps import choropleth_map
from glob import glob

import imageio
import numpy as np
import shutil


def main():

    args = parse_args()

    df1, df2 = parse_data(args)

    df2['dateRep'] = pd.to_datetime(df2['dateRep'], format='%d/%m/%Y')

    # Cumulated sum by country (alpha3).
    df2 = df2[['alpha3', 'dateRep', 'cases', 'deaths']].groupby(['alpha3', 'dateRep']).sum().groupby(level=[0]).cumsum().reset_index()

    # Fill intermittently missing data for each alpha3 group after resampling grouped dataframe.
    df2 = df2.set_index('dateRep').groupby('alpha3').resample('1D').ffill().reset_index(level='dateRep').reset_index(drop=True)

    # df1.rename(columns={'alpha3': 'Code'}, inplace=True)
    # df2.rename(columns={
    #     'alpha3': 'Code',
    #     'cases': 'Value',
    #     }, inplace=True)
    # df1.rename(columns={'admin': 'Country'}, inplace=True)
    # df2['Years'] = pd.Series(range(1, df1.shape[0] + 1))
    # del df2['dateRep']
    # print(df2["Value"].describe())
    # bins = [0, 1, 10, 100, 1000, 10000, 100000]
    # data = df2
    # geo = df1

# Index(['country', 'country_code', 'geometry', 'Entity', 'Year',
#        'Average land surface temperature anomaly',
#        'Average land surface temperature anomaly weighted by population',
#        'Average land surface temperature anomaly weighted by area'],
#       dtype='object')

    df1.rename(columns={'iso_a3': 'alpha3'}, inplace=True)
    # df = pd.merge(df1, df2, on=['alpha3'], how='left').fillna(value={'cases': 0, 'deaths': 0})

    # https://matplotlib.org/examples/color/colormaps_reference.html
    # maxDateRep = max(df2['dateRep'].unique())
    # for boolLog in (False, True,):
    boolLog = True
    cmap = 'OrRd'
    for column in ('cases', 'deaths'):
        # df = pd.merge(df1, df2[df2['dateRep'] == maxDateRep], on=['alpha3'], how='left').fillna(value={'cases': 0, 'deaths': 0})
        # zlim_max = max(10**6 * df[column] / df['pop_est'])
        valuemax = {
        'cases': 10000,  # Italy 1224.1
        'deaths': 1000,  # Italy 123.5
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
        for DateRep in sorted(df2['dateRep'].unique()):
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
                vmin = {'cases': -2, 'deaths': -3}[column]
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
                'CoViD19 {}\n{}'.format(
                    column, dateString), fontsize='large')
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
            path = 'covid19_{}_{}_log{}_{}.png'.format(
                column, cmap, boolLog, dateString,
                )
            # fig.colorbar()
            plt.savefig(path, dpi=100)
            images.append(imageio.imread(path))
            # os.path.remove(path)
            plt.close()

        shutil.copyfile(path, 'covid19_{}_{}_log{}.png'.format(
                column, cmap, boolLog))
        imageio.mimsave('covid19_{}_{}_log{}.gif'.format(column, cmap, boolLog), images, fps=2)

    exit()

    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.div.html
    # https://stackoverflow.com/questions/40892096/divide-two-pandas-dataframes-using-a-dictionary-as-a-key

    # https://cbouy.github.io/2019/06/09/interactive-map.html
    # https://cbouy.github.io/assets/blog/2019/06/09/alcohol-consumption.html
    choropleth_map(
        data, geo,
        bins=bins, bin_labels="L",
        size=(900,450),
        value_title="Litres of pure alcohol per capita",
        value_axis_label="Litres of pure alcohol per capita",
        map_title="COVID19 cases on day {}",
        chart_title="COVID19",
        map_palette="YlGnBu", default_map_year=1,
    )

    exit()

    # hover tool for the map.
    map_hover = HoverTool(tooltips=[ 
        ('Country','@Country (@Code)'),
        ('Obesity rate (%)', '@Prevalence')
    ])

    # button for the animation
    anim_button = Toggle(label="▶ Play", button_type="success", width=50, active=False)

    geosource = GeoJSONDataSource(geojson=df.to_json())

    output_file("covid19map.html", title="CoViD19 cases", mode="inline")

    p = figure(
        title = 'Share of adults who are obese in 1975', 
        plot_height=550 , plot_width=1100, 
        toolbar_location="right", tools="tap,pan,wheel_zoom,box_zoom,save,reset", toolbar_sticky=False,
        active_scroll="wheel_zoom",
    )

    # Add patches (countries) to the figure
    patches = p.patches(
        'xs','ys', source=displayed_src, 
        fill_color='color',
        line_color='black', line_width=0.25, fill_alpha=1, 
        hover_fill_color='color',
    )
    return


def parse_data(args):

    # domain = 'https://www.ecdc.europa.eu'
    # basename = 'COVID-19-geographic-disbtribution-worldwide-{}.xlsx'.format(
    #     args.date)
    # url = '{}/sites/default/files/documents/{}'.format(domain, basename)
    # df_covid19 = parse_url(url)
    df_covid19 = pd.read_csv('../csv')

    df_covid19.rename(columns={
        'geoId': 'alpha2',
        'countryterritoryCode': 'alpha3',
        }, inplace=True)

    # # Get ISO 3166-1 alpha-2 and alpha-3 codes for each country.
    # df_iso3166 = pd.DataFrame(iso3166.countries)

    # url = 'https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/110m/cultural/ne_110m_admin_0_countries.zip'
    # path = download_url(url)
    # with zipfile.ZipFile(path, 'r') as f:
    #     f.extractall()
    # # Read shape file with polygons for each country.
    # df_geo = gpd.read_file(path[:-3] + 'shp')[['ADMIN', 'ADM0_A3', 'geometry']]
    # # Rename column headers.
    # df_geo.columns = ['admin', 'alpha3', 'geometry']
    # # Get rid of Antarctica, because it takes up space.
    # df_geo.drop(df_geo[df_geo['admin'] == 'Antarctica'].index, inplace=True)

    # # Merge geo data with alpha2 codes on alpha3 codes.
    # df1 = pd.merge(df_geo, df_iso3166, on=['alpha3'])

# grep France ne_110m_admin_0_countries.README.html
# <p>Countries distinguish between metropolitan (homeland) and independent and semi-independent portions of sovereign states. If you want to see the dependent overseas regions broken out (like in ISO codes, see France for example), use <a href="https://www.naturalearthdata.com/downloads/10m-political-vectors/10m-admin-0-nitty-gritty/">map units</a> instead.</p>

# http://www.naturalearthdata.com/downloads/110m-cultural-vectors/110m-admin-0-countries/
# Countries distinguish between metropolitan (homeland) and independent and semi-independent portions of sovereign states. If you want to see the dependent overseas regions broken out (like in ISO codes, see France for example), use map units instead.

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
    # df_covid19.loc[df_covid19['alpha3'] == 'Somaliland', 'alpha3'] = 'SOM'
    # exit()

    # # Merge covid19 data with alpha3 codes on alpha2 codes.
    # df2 = pd.merge(df_covid19, df_iso3166, on=['alpha2'])

    df2 = df_covid19

    # # Merge geo data with covid19 data on alpha2 codes.
    # df = pd.merge(df, df_covid19.rename(columns={'GeoId': 'alpha2'}), on=['alpha2'])[['admin', 'alpha3', 'cases', 'deaths', 'dateRep']]

    return df1, df2


def download_url(url, path=None):

    if path is None:
        path = os.path.basename(url)

    if not os.path.isfile(path):
        r = requests.get(url)
        if r.status_code != 200:
            print(url)
            exit()
        with open(path, 'wb') as f:
            f.write(r.content)

    return path


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