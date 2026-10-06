# Tommy Carstensen, March 2020
# No copyright. Code is in the public domain.
# Feel free to modify and republish it as you see fit.

import argparse
import itertools
import operator
import os
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from countryinfo import CountryInfo
from scipy.optimize import curve_fit

import regions

# ECDC's weekly cases and deaths, up to ISO week 2021-01. ECDC has withdrawn the URL this came from (https://opendata.ecdc.europa.eu/covid19/casedistribution/csv/), which now serves another file, so this local copy is read instead.
ECDC = 'data/ecdc.csv'

# Whether to fit a logistic curve to each country's cumulative counts. Off since the epidemics' long tails made the fits meaningless.
FIT = False

# The continent of each part of regions.py, for the places countryinfo does not know.
PART2CONTINENT = {
    'EU': 'Europe', 'EuropeNorth': 'Europe', 'EuropeSouth': 'Europe', 'EuropeWest': 'Europe',
    'EuropeEastCentral': 'Europe', 'Nordic': 'Europe',
    'AmericaNorth': 'Americas', 'AmericaSouth': 'Americas', 'AmericaCentral': 'Americas', 'Caribbean': 'Americas',
    'AsiaSouthEast': 'Asia', 'AsiaCentral': 'Asia', 'AsiaEast': 'Asia', 'AsiaSouth': 'Asia', 'AsiaWestern': 'Asia',
    'AfricaNorth': 'Africa', 'AfricaEast': 'Africa', 'AfricaCentral': 'Africa', 'AfricaSouth': 'Africa',
    'AfricaWest': 'Africa',
    'Oceania': 'Oceania',
}


def main():

    args = parseArgs()

    df0 = pd.read_csv(ECDC)
    df0['dateRep'] = pd.to_datetime(df0['dateRep'], format='%d/%m/%Y')
    df0['countriesAndTerritories'] = df0['countriesAndTerritories'].str.replace('_', ' ')
    if args.countries is None:
        args.countries = list(df0['countriesAndTerritories'].unique())

    doCountry2Continent(args, df0)

    # doHeatMapsBSG(args)

    df0 = sumDataFrameAcrossRegion(args, df0)
    # The regions' populations, now that they have rows of their own.
    args.d_country2pop = country2pop(df0)

    doBarPlots(args, df0)

    doHeatMaps(args, df0)

    if not os.path.isfile('scatter_EU_cases.png'):
        doScatterPlots(args, df0)

    if not os.path.isfile('days100_cases_perCapitaFalse_EU.png'):
        for country in args.d_region2countries['website']:
            doLinePlots(args, df0, country, comparison=True)
        for region in args.d_region2countries:
            doLinePlots(args, df0, region, comparison=False)

    doFitPlots(args, df0)


def doBarPlots(args, df0):

    for countriesAndTerritories in df0['countriesAndTerritories'].unique():
        if df0[df0['countriesAndTerritories'] == countriesAndTerritories]['cases_weekly'].sum() < 1000:
            continue
        path = 'plot_bar_{}.png'.format(countriesAndTerritories.replace(' ', '_'))
        print(path)
        if os.path.isfile(path):
            continue
        fig, ax = plt.subplots()
        series = []
        colors = ['#66c2a5', '#fc8d62', '#8da0cb']
        rows = df0.loc[df0['countriesAndTerritories'] == countriesAndTerritories].sort_values(by='dateRep', ascending=True).set_index('dateRep')
        x = list(range(len(rows)))
        for i, k in enumerate(('cases_weekly', 'deaths_weekly')):
            # Do clip to avoid negative values such as the UK:
            # https://www.theguardian.com/world/2020/aug/12/coronavirus-death-toll-in-england-revised-down-by-more-than-5000
            ps = rows[k].clip(lower=0)
            series.append(ps)
            if k == 'deaths_weekly':
                y = [-_ for _ in ps]
            else:
                y = ps
            ax.bar(x, y, label=k[0].upper() + k[1:].replace('_', ' '), color=colors[i])

        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel('Deaths / Cases', color=colors[2])
        cases, deaths = (ps.to_numpy() for ps in series)
        ax2.plot([d / c if c > 100 else np.nan for c, d in zip(cases, deaths, strict=True)], label='Deaths / Cases')
        ax2.tick_params(axis='y', labelcolor=colors[2])
        # ax2.set_xlim(0, ax2.get_xlim()[1])

        ticks = ax.xaxis.get_ticklocs()
        # ticklabels = [l.get_text() for l in ax.xaxis.get_ticklabels()]
        # ticklabels = [item.strftime('%b %d') for item in ticklabels]
        ticklabels = [day.strftime('%b %d') for day in rows.index]
        ticks_modified = []
        ticklabels_modified = []
        for tick in ticks:
            if tick not in x:
                continue
            ticks_modified.append(tick)
            ticklabels_modified.append(ticklabels[x.index(tick)])
        ax.xaxis.set_ticks(ticks_modified)
        ax.xaxis.set_ticklabels(ticklabels_modified, rotation=45, fontsize='x-small')
        ax.set_title(f'{countriesAndTerritories}')
        ax.legend()
        # ax2.legend()
        fig.set_size_inches(16 / 2, 9 / 2)
        fig.savefig(path, dpi=75)
        print(path)
        plt.savefig(path[:-4] + '_thumb.png', dpi=25)
        fig.clf()
        plt.close(fig)


def doHeatMapsBSG(args):

    url = 'https://github.com/OxCGRT/covid-policy-tracker/raw/master/data/OxCGRT_latest.csv'
    path = 'data/bsg.csv'
    df0 = download_and_read(url, path, pd.read_csv)

    for region in args.d_region2countries:
        for k in ('EconomicSupportIndex', 'ContainmentHealthIndex', 'GovernmentResponseIndex', 'StringencyIndex'):
            path = f'plot_heat_bsg_{k}_{region}.png'
            if os.path.isfile(path):
                continue
            lol = []
            countries = []
            for country in sorted(set(args.d_region2countries[region])):
                # print(region, k, country)
                values = df0[df0['CountryName'].isin([country])][k].tolist()
                if len(values) == 0:
                    continue
                if country not in args.d_country2pop:
                    continue
                # if pop < 1:
                # # if pop < 1 and country not in ('Iceland', 'Faroe Islands'):
                #     continue
                country = country.replace('United States of America', 'US')
                countries.append(country)
                # Do max to avoid negative values such as the UK:
                # https://www.theguardian.com/world/2020/aug/12/coronavirus-death-toll-in-england-revised-down-by-more-than-5000
                lol.append([max(0, _) for _ in reversed(values)])
            # array = np.array([np.array(l) for l in lol])
            length = max(map(len, lol))
            array = np.array(list(reversed([list(reversed(xi + [0] * (length - len(xi)))) for xi in lol])))

            fig, ax = plt.subplots()
            fig.set_size_inches(16 / 2, 9 / 2)
            heatmap = ax.pcolor(array, cmap='OrRd')
            plt.colorbar(heatmap)
            ax.set_yticks(np.arange(array.shape[0]) + 0.5, minor=False)
            ax.set_yticklabels(list(reversed(countries)), minor=False, fontsize='x-small')
            ax.set_xlabel('Day')
            ax.set_title(f'{region}\n{k[0].upper()}{k[1:]}')
            fig.set_layout_engine('tight')
            plt.savefig(path, dpi=75)
            print(path)
            plt.clf()
            plt.close()
            # im = ax.imshow(array)


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


def doHeatMaps(args, df0):

    for region in args.d_region2countries:
        for k in ('cases', 'deaths'):
            path = f'plot_heat_{k}_{region}.png'
            if os.path.isfile(path):
                continue
            k += '_weekly'
            lol = []
            countries = []
            for country in sorted(set(args.d_region2countries[region])):
                # print(region, k, country)
                # Use clip to avoid negative counts of cases and/or deaths.
                values = df0[df0['countriesAndTerritories'].isin([country])][k].clip(lower=0).rolling(window=7, min_periods=1).mean().to_list()
                if len(values) == 0:
                    continue
                try:
                    pop = args.d_country2pop[country]
                except KeyError:
                    continue
                # if pop < 1 and country not in ('Iceland', 'Faroe Islands'):
                if pop < 1:
                    continue
                country = country.replace('United States of America', 'US')
                countries.append(country)
                lol.append([_ / pop for _ in values])
            # array = np.array([np.array(l) for l in lol])
            length = max(map(len, lol))
            array = np.array(list(reversed([list(reversed(xi + [0] * (length - len(xi)))) for xi in lol])))

            fig, ax = plt.subplots()
            heatmap = ax.pcolor(array, cmap='OrRd')
            plt.colorbar(heatmap)
            ax.set_yticks(np.arange(array.shape[0]) + 0.5, minor=False)
            ax.set_yticklabels(list(reversed(countries)), minor=False, fontsize='x-small')
            ax.set_xlabel('Week')
            ax.set_title('{}\n{}{} per million'.format(region, k[0].upper(), k[1:].replace('_', ' ')))
            fig.set_layout_engine('tight')
            plt.savefig(path, dpi=75)
            print(path)
            plt.clf()
            plt.close()
            # im = ax.imshow(array)


def doFitPlots(args, df0):

    df = (
        df0[df0['countriesAndTerritories'].isin(args.countries)]
        .filter(['cases_weekly', 'dateRep', 'deaths_weekly'])
        .groupby('dateRep').sum())
    print(df.tail(1))

    # Exclude the most recent data point, which does not capture all new cases.
    yConfCasesCumYesterday = np.delete(df['cases_weekly'].values.cumsum(), -1)

    # Less than x cases.
    # if max(yConfCasesCumYesterday) < 5 and len(
    if max(yConfCasesCumYesterday) < 1000 and len({
            'Singapore', 'Taiwan', 'Hong Kong', 'Japan',
            'United States of America', 'EU', 'China',
            'Germany', 'India', 'United Kingdom', 'France', 'Italy',
            'Brazil', 'Canada', 'South Korea', 'Spain', 'Australia', 'Mexico',
            'Indonesia', 'Netherlands', 'Saudia Arabia', 'Turkey', 'Switzerland',
            'Peru',
            # 'Russia',
            } & set(args.countries)) == 0:  # Singapore 187
        print('Insufficient cumulated cases (n={}) to carry out fitting.'.format(df['cases_weekly'].values.sum()))
        x = df.index.strftime('%Y-%m-%d').values
        y = df['cases_weekly'].values.cumsum()
        z = df['deaths_weekly'].values.cumsum()
        print('\n'.join('{}\t{}\t{}'.format(*t) for t in zip(x, y, z)))
        sys.exit()

    # # Skip if less than 40 days since first case.
    # # Midpoint of China curve was after 40 days.
    # if operator.sub(
    #     len(yConfCasesCumYesterday),
    #     list(yConfCasesCumYesterday).count(0),
    #     ) < 40 and len(
    #         # if max(yConfCasesCumYesterday) < 100 and len(
    #         set([
    #         'Singapore', 'Taiwan', 'Hong Kong', 'Japan', 'Iran',
    #         'United States of America', 'EU', 'China',
    #         'Japan', 'Germany', 'India', 'United Kingdom', 'France', 'Italy',
    #         'Brazil', 'Canada', 'Russia', 'South Korea', 'Spain', 'Australia', 'Mexico',
    #         'Indonesia', 'Netherlands', 'Saudia Arabia', 'Turkey', 'Switzerland',
    #         ]) & set(args.countries)) == 0:
    #     # ) < 1 * 7:
    #     print('Only {} days have passed. {} confirmed cases today. Exiting.'.format(
    #         operator.sub(
    #             len(yConfCasesCumYesterday),
    #             list(yConfCasesCumYesterday).count(0)),
    #         df['cases'].values.sum(),
    #         ))
    #     exit()

    colors = define_colors()

    for k in ('cases_weekly', 'deaths_weekly'):

        plot_per_country(args, df, k, colors)


def sumDataFrameAcrossRegion(args, df0: pd.DataFrame) -> pd.DataFrame:

    for region, countries in args.d_region2countries.items():
        rows = df0.loc[df0['countriesAndTerritories'].isin(countries)]
        df = (
            rows.filter(['cases_weekly', 'dateRep', 'deaths_weekly'])
            .groupby('dateRep').sum().reset_index())
        df['countriesAndTerritories'] = region
        df['popData2019'] = rows.drop_duplicates('countriesAndTerritories')['popData2019'].sum()
        df0 = pd.concat([df0, df])

    return df0


def country2pop(df0):
    """Each country's population in millions, ECDC's of 2019, as the maps on the page use."""
    rows = df0.dropna(subset=['popData2019']).drop_duplicates('countriesAndTerritories')
    return dict(zip(rows['countriesAndTerritories'], rows['popData2019'] / 10**6, strict=True))


def doCountry2Continent(args, df0):
    """Each country's continent, from countryinfo, and its population, from ECDC. A place countryinfo does not know takes the continent of its part in regions.py."""

    args.d_country2pop = country2pop(df0)
    args.d_country2continent = {}
    args.d_country2continent['EU'] = 'Europe'
    codes = set(df0['countryterritoryCode'].unique())
    for d in CountryInfo().all().values():
        if 'region' not in d:
            continue
        # args.d_country2continent[d['name']] = d['subregion']
        args.d_country2continent[d['name']] = d['region']
        if d['ISO']['alpha3'] not in codes:
            continue
        # ECDC's name for the country, where it differs from countryinfo's.
        country = df0[df0['countryterritoryCode'] == d['ISO']['alpha3']]['countriesAndTerritories'].unique()[0]
        args.d_country2continent[country] = d['region']

    # countryinfo's region for these is not the one of their neighbours.
    args.d_country2continent['United States of America'] = 'Americas'
    args.d_country2continent['Bahamas'] = 'Americas'

    for part, countries in regions.PARTS.items():
        for country in countries:
            args.d_country2continent.setdefault(country.replace('_', ' '), PART2CONTINENT[part])


def define_colors():

    # https://colorbrewer2.org/#type=qualitative&scheme=Paired&n=4
    colors = ('#a6cee3', '#1f78b4', '#b2df8a', '#33a02c')
    # https://colorbrewer2.org/#type=qualitative&scheme=Paired&n=3
    colors = ('#a6cee3', '#1f78b4', '#b2df8a')
    # https://colorbrewer2.org/#type=qualitative&scheme=Dark2&n=3
    colors = ('#1b9e77', '#d95f02', '#7570b3')
    # https://colorbrewer2.org/#type=qualitative&scheme=Set1&n=3
    colors = ('#e41a1c', '#377eb8', '#4daf4a')
    # https://colorbrewer2.org/#type=qualitative&scheme=Set2&n=3
    colors = ('#66c2a5', '#fc8d62', '#8da0cb')

    return colors


def plot_per_country(args, df, k, colors):

    colorFit = colors[0]
    colorNewCases = colors[1]
    colorErr = colors[2]

    yCumYesterday = np.delete(df['cases_weekly'].values.cumsum(), -1)
    dayFirstCase = yCumYesterday.tolist().count(0)

    booleans = (
        any((
            # Cases today less than yesterday.
            df[k].values[-1] < df[k].values[-2],
            # Cases today less than 10.
            df[k].values[-1] < 10,
            # Cases today less than twenty percent of maximum.
            df[k].values[-1] < 0.20 * max(df[k].values)
            )),

        # Cases today and yesterday less than a percentage of maximum.
        all((
            df[k].values[-1] < 0.85 * max(df[k].values),
            df[k].values[-2] < 0.85 * max(df[k].values),
            )),

        # At least a number of days since the first case must have passed.
        operator.sub(
            len(yCumYesterday),
            list(yCumYesterday).count(0),
            ) > 50,

        # At least a number of cases must have been confirmed.
        max(yCumYesterday) > 1000,
        )
    tFit = None
    if FIT and (all(booleans) or len({
            'United States of America',
            'United Kingdom',
            'Germany',
            'Italy',
            'South Korea',
            'France',
            'Japan',
            'Peru',
            } & set(args.countries)) > 1):
        yFit = np.delete(df[k].values.cumsum(), -1)
        xFit = list(range(len(yFit)))
        tFit = fit(args, df, xFit, yFit)
    else:
        print(yCumYesterday)
        print(list(yCumYesterday).count(0))
        print(len(yCumYesterday))
        print(operator.sub(
            len(yCumYesterday),
            list(yCumYesterday).count(0),
            ))
        print(booleans)
        print(df[k].values[-1] / max(df[k].values))
        print(df[k].values[-2] / max(df[k].values))
        print(max(df[k].values))
        # exit()

    # popt[0] = max(popt[0], df['cases'].values.sum())
    # Do not add fit to plot, if calculated maximum is greater than actual maximum.
    if tFit is not None and tFit[0][0] > 1.05 * max(yCumYesterday):
        tFit = None

    plt.xlabel('Weeks')
    plt.ylabel(k[0].upper() + k[1:].replace('_', ' '))

    if tFit is not None:
        popt, perr = tFit
        xFit = list(range(2 * len(yCumYesterday)))
        plt.plot(
            xFit,
            logistic(xFit, *popt),
            color=colorFit,
            label='Fitted logistic function',
            zorder=2,
            linewidth=4,
            )
        for _ in range(1000):
            a = random.triangular(popt[0] - 3 * perr[0], popt[0] + 3 * perr[0])
            b = random.triangular(popt[1] - 3 * perr[1], popt[1] + 3 * perr[1])
            c = random.triangular(popt[2] - 3 * perr[2], popt[2] + 3 * perr[2])
            # Skip if calculated maximum is smaller than actual cumulated cases.
            if 1.1 * a < df[k].values.sum():
                continue
            assert a > 0
            # Calculated cases plus 10 percent should be greated than actual cases.
            assert 1.1 * a >= max(yCumYesterday)
            assert b > 0
            assert b < 2, (b, popt[1])
            assert c > 0
            assert c < 2 * len(yCumYesterday)
            yFit = [logistic(x, a, b, c) for x in xFit]
            plt.plot(xFit, yFit, colorErr, alpha=.05, zorder=1)

        # The legend's entry for the faint curves.
        plt.plot(
            [],
            [],
            color=colorErr,
            label='Other possible outcomes',
            zorder=1,
            )

    plt.scatter(
        list(range(len(df))),
        df[k].values.cumsum(),
        color='black',
        label=f'Cumulated {k}',
        zorder=3,
        )

    plt.bar(
        list(range(len(df))),
        df[k].values,
        color=colorNewCases,
        label=f'New {k}',
        zorder=4,
        )

    title = args.title
    title += ', ' + k[0].upper() + k[1:]
    # try:
    #     title += ' population={:.1f}M'.format(d_country2pop[args.title])
    # except KeyError:
    #     pass
    title += f', {args.dateToday}'
    title += '\nCases this week={}, Deaths this week={}'.format(
        df['cases_weekly'].values[-1],
        df['deaths_weekly'].values[-1],
        )
    if tFit is not None:
        popt = tFit[0]
        title += f'\nCalculated cumulated {k}={int(popt[0]):d}, midpoint={int(popt[2]):d}, steepness={popt[1]:.2f}'
    title += f'\nCurrent day={len(df)}'
    title += f', Day of first case={dayFirstCase}'
    title += ', Total confirmed cases={}'.format(max(df['cases_weekly'].values.cumsum()))
    title += ', Total confirmed deaths={}'.format(int(max(df['deaths_weekly'].values.cumsum())))
    plt.title(title, fontsize='x-small')
    plt.legend()
    # plt.yscale('log')
    path = f'COVID19_sigmoid_{k}_{args.affix}_{args.dateToday}.png'
    print(path)
    plt.savefig(path, dpi=75)
    path = f'COVID19_sigmoid_{k}_{args.affix}.png'
    print(path)
    plt.savefig(path, dpi=75)
    print(path)
    plt.clf()

    if k == 'cases_weekly':  # just do table once for cases

        try:
            popSize = args.d_country2pop[args.title]
        except KeyError:
            return
        s = '<tr>'
        s += f'<td>{args.title}</td>'
        s += f'<td>{popSize:.1f}</td>'
        s += f'<td>{args.d_country2continent.get(args.title)}</td>'
        s += '<td>{}</td>'.format(df['cases_weekly'].values.sum())
        s += f'<td><a href="days100_cases_perCapitaFalse_{args.affix}.png"><img src="days100_cases_perCapitaFalse_{args.affix}_thumb.png" height="45"></a></td>'
        # s += '<td><a href="plot_bar_cases_{}.png"><img src="plot_bar_cases_{}_thumb.png" height="45"></a></td>'.format(args.affix, args.affix)
        s += '<td>{}</td>'.format(int(df['deaths_weekly'].values.sum()))
        s += f'<td><a href="days100_deaths_perCapitaFalse_{args.affix}.png"><img src="days100_deaths_perCapitaFalse_{args.affix}_thumb.png" height="45"></a></td>'
        # s += '<td><a href="plot_bar_deaths_{}.png"><img src="plot_bar_deaths_{}_thumb.png" height="45"></a></td>'.format(args.affix, args.affix)
        s += f'<td><a href="plot_bar_{args.affix}.png"><img src="plot_bar_cases_{args.affix}_thumb.png" height="45"></a></td>'
        s += '<td>{:.1f}</td>'.format(100 * df['deaths_weekly'].values.sum() / df['cases_weekly'].values.sum())
        s += '<td>{}</td>'.format(df['cases_weekly'].values[-1])
        s += '<td>{}</td>'.format(df['deaths_weekly'].values[-1])
        try:
            s += '<td>{:.1f}</td>'.format(df['cases_weekly'].values.sum() / popSize)
            s += '<td>{:.1f}</td>'.format(df['deaths_weekly'].values.sum() / popSize)
        except UnboundLocalError:
            pass

        # https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv
        df_owid = pd.read_csv('data/owid.csv')
        if args.title == 'United States of America':
            location = 'United States'
            v = df_owid[df_owid['location'] == location]['total_tests_per_thousand'].max()
        elif args.title == 'EU':
            # Calculate weighted average.
            nominator = 0
            denominator = 0
            for location in args.d_region2countries['EU']:
                v1 = df_owid[df_owid['location'] == location]['total_tests_per_thousand'].max()
                v2 = df_owid[df_owid['location'] == location]['total_tests'].max()
                if np.isnan(v1):
                    continue
                denominator += v2 / v1
                nominator += v2
            v = nominator / denominator
        else:
            location = args.title
            v = df_owid[df_owid['location'] == location]['total_tests_per_thousand'].max()
        if np.isnan(v):
            return
        s += f'<td>{v:.1f}</td>'

        s += '</tr>'
        with open(f'table{args.affix}.txt', 'w') as f:
            print(s, file=f)

    return


def doScatterPlots(args, df0):

    for region in args.d_region2countries:
        for k in ('cases', 'deaths'):
            k += '_weekly'
            d = {k: ([], []) for k in set(args.d_country2continent.values())}
            xx = []
            yy = []
            for country in args.d_region2countries[region]:
                # print(region, k, country)
                try:
                    x = 10**6 * args.d_country2pop[country]
                except KeyError:
                    continue
                y = df0[df0['countriesAndTerritories'].isin([country])][k].max()
                try:
                    label = args.d_country2continent[country]
                except KeyError:
                    continue
                # Greenland
                if label == 'Northern America' and region == 'Europe':
                    continue  # Greenland
                d[label][0].append(x)
                d[label][1].append(y)
                plt.annotate(country, (x, y), fontsize='xx-small')
                if y > 0:
                    xx.append(x)
                    yy.append(y)
            print(region, np.corrcoef(xx, yy)[0][1])
            for label, t in d.items():
                if len(t[0]) == 0:
                    continue
                plt.scatter(
                    t[0], t[1],
                    label=label,
                    linewidth=2,
                    )
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Population size')
            plt.ylabel(k[0].upper() + k[1:].replace('_weekly', '').replace('_', ' '))
            plt.legend(prop={'size': 6})
            plt.title(region + '\n' + k[0].upper() + k[1:].replace('_weekly', '').replace('_', ' '))
            # plt.label()
            path = 'scatter_{}_{}.png'.format(region, k.replace('_weekly', ''))
            plt.savefig(path, dpi=75)
            plt.clf()


def doLinePlots(args, df0, key_geo, comparison=True):

    if comparison:
        region = 'WorldAll'
        if key_geo == 'EU':
            return
    else:
        region = key_geo

    for perCapita in (True, False,):
        # for k, limSum in (('cases_weekly', 20000), ('deaths_weekly', 500)):
        for k in ('cases', 'deaths'):
            print('line', key_geo, perCapita, k)

            path = 'days100_{}_perCapita{}_{}.png'.format(k, perCapita, key_geo.replace(' ', '_'))
            if os.path.isfile(path):
                continue

            k += '_weekly'

            d = {True: [], False: []}
            # for country in df0['countriesAndTerritories'].unique():
            labels = set()
            for country in args.d_region2countries[region]:
                if comparison is True and country == key_geo:
                    continue
                if df0[df0['countriesAndTerritories'].isin([country])][k].sum() == 0:
                    continue
                if perCapita is True:
                    value = operator.truediv(
                        10**6 * df0[df0['countriesAndTerritories'].isin([country])][k].sum(),
                        df0[df0['countriesAndTerritories'].isin([country])]['popData2019'].unique(),
                        )[0]
                    if np.isnan(value):
                        continue
                else:
                    value = df0[df0['countriesAndTerritories'].isin([country])][k].sum()
                if comparison is True:
                    _ = args.d_country2continent[country] == args.d_country2continent[key_geo]
                else:
                    _ = True
                d[_].append((value, country))

            if comparison is True:
                tuples = itertools.chain(
                    sorted(d[False], reverse=True),
                    sorted(d[True], reverse=True),
                    [(None, key_geo)],
                    )
            else:
                tuples = itertools.chain(
                    sorted(d[False], reverse=True),
                    sorted(d[True], reverse=True),
                    )

            lim = 1 if perCapita else {'cases_weekly': 1000, 'deaths_weekly': 100}[k]
            for t in tuples:
                country = t[1]
                df = df0[df0['countriesAndTerritories'].isin([country])].sort_values(by='dateRep', ascending=True)
                if k == 'deaths_weekly' and 'South' in country:
                    print(k, country, df[k].sum())
                if df[k].sum() < 10:
                    continue
                # if df[k].sum() < limSum and country not in (
                #     'Japan', 'South_Korea', 'Taiwan', 'Singapore',
                #     'United_States_of_America', 'United_Kingdom',
                #     ):
                #     continue
                # if country in ('Iran',):
                #     continue
                # print(country, df[k].sum())
                # print(country, df[k].sum())
                if perCapita is False:
                    y = df[k].cumsum()[df[k].cumsum() > lim].values
                else:
                    # s = 10**6 * df[k].cumsum()[df[k].cumsum() > 100].values / df['popData2018'].unique()[0]
                    # y = s
                    s = 10**6 * df[k].cumsum() / df['popData2019'].unique()
                    y = s[s > lim]
                    y = df[k].cumsum()[df[k].cumsum() > 100].values / df['popData2019'].unique()
                if df[k].sum() < lim:
                    continue
                if len(y) == 0:
                    continue
                x = list(range(len(y)))
                if comparison is False:
                    color = None
                    label = '{} ({:d})'.format(
                        country.replace('_', ' ').replace('United States of America', 'US').replace('United Kingdom', 'UK'),
                        int(max(y)),
                        )
                    linewidth = 2
                else:
                    if country == key_geo:
                        # country = country.replace('United States of America', 'US')
                        # country = country.replace('United Kingdom', 'UK')
                        color = '#e41a1c'  # red
                        label = '{} ({:d})'.format(
                            country.replace('_', ' '),
                            int(max(y)),
                            )
                        linewidth = 4
                    else:
                        continent = args.d_country2continent[country]
                        color = {
                            # 'North America': '#8dd3c7',
                            'Americas': '#8dd3c7',
                            'Africa': '#ffffb3',
                            'Europe': '#bebada',
                            'Asia': '#fb8072',
                            'Oceania': '#80b1d3',
                            }[continent]

                        if continent == args.d_country2continent[key_geo]:
                            color = 'darkgrey'
                            label = continent
                        else:
                            color = 'lightgrey'
                            label = 'Rest of World'

                        if label in labels:
                            label = None
                        else:
                            labels.add(label)

                        linewidth = 2

                plt.semilogy(
                    x, y,
                    label=label,
                    color=color,
                    linewidth=linewidth,
                    )
            plt.legend(prop={'size': 6})

            if perCapita is True:
                textPerCapita = ' per 1 million capita'
            else:
                textPerCapita = ''

            if lim == 1:
                kSingPlur = k.lower().split('_')[0][:-1]
            else:
                kSingPlur = k.lower().split('_')[0]

            plt.xlabel(f'Weeks since {lim} confirmed {kSingPlur}{textPerCapita}')
            plt.ylabel('Cumulated confirmed {}{}'.format(k.lower().split('_')[0], textPerCapita))
            if lim == 1:
                textLim = f'{k.lower()[:-1]}{textPerCapita}'
                textLim = kSingPlur
            else:
                textLim = f'{k.lower()}{textPerCapita}'
                textLim = kSingPlur
            keyUpperCase = '{}{} {}'.format(k.split('_')[1][0].upper(), k.split('_')[1][1:], k.split('_')[0])
            plt.title(f'{key_geo}\n{keyUpperCase}{textPerCapita} after first week with more than {lim} {textLim}', fontsize='small')
            plt.savefig(path, dpi=75)
            plt.savefig(path[:-4] + '_thumb.png', dpi=25)
            plt.clf()

    return


def fit(args, df, xCumYesterday, yCumYesterday):

    # Seed values for regression. Heuristic approach.
    if 0 in yCumYesterday:
        guessMidpoint = min(60, 40 + list(yCumYesterday[::-1]).index(0))
    else:
        guessMidpoint = 40
    # Cases less than 5 percent of ConfCasesCum
    if df['cases_weekly'].values[-2] / yCumYesterday[-1] < 0.05:
        guessCasesMax = 1.5 * max(yCumYesterday)
    else:
        guessCasesMax = 10 * max(yCumYesterday)
    print('guessMidpoint', guessMidpoint)
    print('guessCasesMax', guessCasesMax)
    # guessCasesMax = 100000
    # guessMidpoint = 76
    p0 = [guessCasesMax, 0.5, guessMidpoint]
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
    try:
        popt, pcov = curve_fit(
            logistic,
            xCumYesterday,
            yCumYesterday,
            p0=p0,
            )
        perr = np.sqrt(np.diag(pcov))
        print('maximum', popt[0])
        print('midpoint', popt[2] - 3 * perr[2], popt[2] + 3 * perr[2])
        print('steepness', popt[1])
    except RuntimeError:
        print('Fitting failed')
        print('\n'.join('{}\t{}\t{}'.format(*t) for t in zip(
            xCumYesterday, yCumYesterday,
            map(int, df['deaths_weekly'].values.cumsum()),
            strict=False,
            )))
        return None
    except (ValueError, TypeError):
        print('Exception')
        print('\n'.join('{}\t{}\t{}'.format(*t) for t in zip(
            xCumYesterday, yCumYesterday,
            df['deaths_weekly'].values.cumsum(),
            strict=False,
            )))
        return None

    # Heuristic checks of whether curve fitting makes sense.
    booleans = (
        # Calculated cases plus 10 percent should be greated than actual cases today.
        1.1 * popt[0] < df['cases_weekly'].values.sum(),
        # Steepness should not exceed 1.
        popt[1] > 1,
        # Error greater than half of calculated or actual maximum.
        3 * perr[0] > .5 * popt[0],
        perr[0] > df['cases_weekly'].values.sum(),
        )

    if any(booleans):
        print('\n'.join('{}\t{}\t{}'.format(*t) for t in zip(
            df.index.strftime('%Y-%m-%d').values,
            df['cases_weekly'].values.cumsum(),
            df['deaths_weekly'].values.cumsum(),
            strict=True,
            )))
        print(f'Fitting yielded unreliable results for {args.title}.')
        print('popt', popt)
        print('perr', perr)
        print('ConfCasesSum', df['cases_weekly'].values.sum())
        print(popt)
        print(booleans)
        return

    return popt, perr


def logistic(x, a, b, c, amax=None):

    # a is maximum
    # b is steepness/inclination
    # c is midpoint

    if amax is not None:
        a = amax

    y = a / (1 + np.exp(-b * (x - c)))

    return y


def names(countries):
    """Sorted, each once, under the names main() gives ECDC's countries (spaces for underscores)."""
    return sorted({country.replace('_', ' ') for country in countries})


def regionLists():
    """The regions and country sets the charts are drawn for. The regions are those of regions.py, which corrects the 2020 lists' spellings (Czechia was 'Czech Republic', so the EU left it out) and adds the places they missed."""

    d_region2countries = {part: names(countries) for part, countries in regions.PARTS.items()}
    d_region2countries.update({region: names(countries) for region, countries in regions.REGIONS.items()})
    d_region2countries['Scandinavia'] = ['Denmark', 'Norway', 'Sweden']
    d_region2countries['AsiaExChina'] = names(set(d_region2countries['Asia']) - {'China'})
    d_region2countries['AsiaEastExChina'] = names(set(d_region2countries['AsiaEast']) - {'China'})
    d_region2countries['AsiaWesternExIran'] = names(set(d_region2countries['AsiaWestern']) - {'Iran'})
    d_region2countries['LatinAmerica'] = names(
        set(d_region2countries['AmericaNorth'] + d_region2countries['AmericaSouth'])
        - {'United States of America', 'Canada'})
    d_region2countries['LatinAmericaExVenezuela'] = names(set(d_region2countries['LatinAmerica']) - {'Venezuela'})
    d_region2countries['AmericaSouthExVenezuela'] = names(set(d_region2countries['AmericaSouth']) - {'Venezuela'})

    d_region2countries['WorldAll'] = names({
        country for countries in d_region2countries.values() for country in countries})

    d_region2countries['World1'] = names({
        'EU',
        'United States of America',
        'Brazil',
        'India',
        'Mexico',
        # 'Italy',
        'United Kingdom',
        'Iran',
        # 'Spain',
        'Russia',
        'Argentina',
        'Colombia',
        'Peru',
        'South Africa',
        # 'Poland',
        'Indonesia',

        # 'China',  # fake numbers?
        # 'Iran',  # fake numbers?
        # 'Singapore',
        # 'Macao',

        'Sweden',
        # 'Estonia',
        })

    d_region2countries['World2'] = names({
        'Taiwan',
        'Vietnam',
        'Thailand',
        'Singapore',
        'New Zealand',
        'South Korea',
        'Malaysia',
        'Japan',
        'Australia',

        # 'China',  # fake numbers?
        # 'Iran',  # fake numbers?
        # 'Singapore',
        'Uruguay',
        # 'Israel',
        # 'Macao',

        # 'Greece',
        'Denmark',
        'Sweden',
        # 'Austria',
        # 'Estonia',

        # Non-EU Europe
        'Iceland',
        'Norway',
        })

    d_region2countries['GoogleTrends'] = names({
        'Germany',
        'Brazil',
        'India',
        'United States of America',
        })

    d_region2countries['website'] = names({
        'United States of America',
        # 'China',  # fake numbers?
        # 'Iran',  # fake numbers?
        'United Kingdom',
        'South Korea',
        'Japan',
        'Singapore',
        'Taiwan',
        'Iceland',
        'Australia',
        'South Africa',
        'Senegal',
        'New Zealand',
        'Norway',
        'Malaysia',
        'Sweden',
        'Hong Kong',
        # 'Macao',
        'EU',
        'Italy',
        'Spain',
        'Germany',
        'France',
        'Denmark',
        'Austria',
        'Estonia',
        'Belgium',
        'Uruguay',
        'Vietnam',
        'Israel',
        'Greece',
        })

    return d_region2countries


def parseArgs():

    d_region2countries = regionLists()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--countries', nargs='*',
        help='e.g. Denmark,Sweden,Norway,Finland,Iceland',
        )
    parser.add_argument('--region', choices=sorted(d_region2countries))
    parser.add_argument(
        '--dateToday', default=time.strftime('%Y-%m-%d'),
        help='Date in ISO 8601 format YYYY-MM-DD',
        required=False,
        )
    args = parser.parse_args()

    print(args)

    if args.countries is not None:
        args.countries = ' '.join(args.countries).split(',')
        args.title = ','.join(_.replace('_', ' ') for _ in args.countries)
        args.affix = ''.join(args.countries).replace(' ', '_')
    elif args.region is not None:
        args.countries = d_region2countries[args.region]
        args.title = args.region
        args.affix = args.region
    else:
        # Every country in the data, which main() fills in once it has read them.
        args.countries = None
        args.title = 'World'
        args.affix = 'World'

    args.d_region2countries = d_region2countries

    return args


if __name__ == '__main__':
    main()
