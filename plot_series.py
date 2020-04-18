# Tommy Carstensen, March 2020
# No copyright. Code is in the public domain.
# Feel free to modify and republish it as you see fit.

from datetime import datetime
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import os
import requests
import random
import operator
from countryinfo import CountryInfo
import itertools


def main():

    args = parseArgs()

    domain = 'https://www.ecdc.europa.eu'
    basename = 'COVID-19-geographic-disbtribution-worldwide-{}.xlsx'.format(
        args.dateToday)
    url = '{}/sites/default/files/documents/{}'.format(domain, basename)
    url = 'https://opendata.ecdc.europa.eu/covid19/casedistribution/csv'
    df0 = parseURL(url)

    df0['dateRep'] = pd.to_datetime(df0['dateRep'], format='%d/%m/%Y')
    df0['countriesAndTerritories'] = df0['countriesAndTerritories'].str.replace('_', ' ')

    # Get population sizes.
    args.d_country2continent = {'United States of America': 'North America'}
    for country, d in CountryInfo().all().items():
        # print(country, d)
        try:
            args.d_country2pop[d['name']] = d['population'] / 10**6
            args.d_country2continent[d['name']] = d['subregion']
        except KeyError:
            pass
        if not d['ISO']['alpha3'] in df0['countryterritoryCode'].unique():
            continue
        country = df0[df0['countryterritoryCode'] == d['ISO']['alpha3']]['countriesAndTerritories'].unique()[0]
        try:
            args.d_country2pop[country] = d['population'] / 10**6
            args.d_country2continent[d['name']] = d['subregion']
        except KeyError:
            continue

    if not os.path.isfile('scatter_EU_cases.png'):
        doScatterPlots(args, df0)

    if not os.path.isfile('days100_cases_en_perCapitaFalse_EU.png'):
        doLinePlots(args, df0)

    df = (
        df0[df0['countriesAndTerritories'].isin(args.countries)]
        .filter(['cases', 'dateRep', 'deaths'])
        .groupby('dateRep').sum())
    print(df.tail(1))

    # Exclude the most recent data point, which does not capture all new cases.
    # xConfCasesCumToday = list(range(len(yConfCasesCumToday)))
    yConfCasesCumYesterday = np.delete(df['cases'].values.cumsum(), -1)
    xConfCasesCumYesterday = list(range(len(yConfCasesCumYesterday)))

    dayFirstCase = yConfCasesCumYesterday.tolist().count(0)

    # Less than x cases.
    if max(yConfCasesCumYesterday) < 1000 and len(
    # if max(yConfCasesCumYesterday) < 5 and len(
    set([
    'Singapore', 'Taiwan', 'Hong Kong', 'Japan',
    'United States of America', 'EU', 'China',
    'Japan', 'Germany', 'India', 'United Kingdom', 'France', 'Italy',
    'Brazil', 'Canada', 'South Korea', 'Spain', 'Australia', 'Mexico',
    'Indonesia', 'Netherlands', 'Saudia Arabia', 'Turkey', 'Switzerland',
    'Peru',
    # 'Russia',
    ]) & set(args.countries)) == 0:  # Singapore 187
        print('Insufficient cumulated cases (n={}) to carry out fitting.'.format(df['cases'].values.sum()))
        x = df.index.strftime('%Y-%m-%d').values
        y = df['cases'].values.cumsum()
        z = df['deaths'].values.cumsum()
        print('\n'.join('{}\t{}\t{}'.format(*t) for t in zip(x, y, z)))
        exit()

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

    for k in ('cases', 'deaths'):

        plot_per_country(args, df, k, colors)

    return


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

    yConfCasesCumYesterday = np.delete(df['cases'].values.cumsum(), -1)
    xConfCasesCumYesterday = list(range(len(yConfCasesCumYesterday)))
    dayFirstCase = yConfCasesCumYesterday.tolist().count(0)

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
            len(yConfCasesCumYesterday),
            list(yConfCasesCumYesterday).count(0),
            ) > 50,

        # At least a number of cases must have been confirmed.
        max(yConfCasesCumYesterday) > 1000,
        )
    if all(booleans) or len(set((
        'United States of America',
        'United Kingdom',
        'Germany',
        'Italy',
        'South Korea',
        'France',
        'Japan',
        'Peru',
        )) & set(args.countries)) > 1:
        yFit = np.delete(df[k].values.cumsum(), -1)
        xFit = list(range(len(yFit)))
        tFit = fit(args, df, xFit, yFit)
    else:
        tFit = None
        print(yConfCasesCumYesterday)
        print(list(yConfCasesCumYesterday).count(0))
        print(len(yConfCasesCumYesterday))
        print(operator.sub(
            len(yConfCasesCumYesterday),
            list(yConfCasesCumYesterday).count(0),
            ))
        print(booleans)
        print(df[k].values[-1] / max(df[k].values))
        print(df[k].values[-2] / max(df[k].values))
        print(max(df[k].values))
        # exit()

    tFit = None

    if tFit is not None:
        popt, perr = tFit
        # popt[0] = max(popt[0], df['cases'].values.sum())
        fitConfCasesMax, fitSteep, fitMid = popt

    plt.xlabel('Days')
    plt.ylabel(k[0].upper() + k[1:])

    if tFit is not None:
        xFit = list(range(2 * len(yConfCasesCumYesterday)))
        plt.plot(
            xFit,
            logistic(xFit, *popt),
            color=colorFit,
            label='Fitted logistic function',
            zorder=2,
            linewidth=4,
            )
        for i in range(1000):
            a = random.triangular(popt[0] - 3 * perr[0], popt[0] + 3 * perr[0])
            b = random.triangular(popt[1] - 3 * perr[1], popt[1] + 3 * perr[1])
            c = random.triangular(popt[2] - 3 * perr[2], popt[2] + 3 * perr[2])
            # Skip if calculated maximum is smaller than actual cumulated cases.
            if 1.1 * a < df[k].values.sum():
                continue
            assert a > 0
            # Calculated cases plus 10 percent should be greated than actual cases.
            assert 1.1 * a >= max(yConfCasesCumYesterday)
            assert b > 0
            assert b < 2, (b, popt[1])
            assert c > 0
            assert c < 2 * len(yConfCasesCumYesterday)
            yFit = [logistic(_, a, b, c) for _ in xFit]
            plt.plot(xFit, yFit, colorErr, alpha=.05, zorder=1)

        plt.plot(
            xFit,
            yFit,
            color=colorErr,
            label='Other possible outcomes',
            zorder=1,
            )

    plt.scatter(
        list(range(len(df))),
        df[k].values.cumsum(),
        color='black',
        label='Cumulated {}'.format(k),
        zorder=3,
        )

    plt.bar(
        list(range(len(df))),
        df[k].values,
        color=colorNewCases,
        label='New {}'.format(k),
        zorder=4,
        )

    title = args.title
    title += ', ' + k[0].upper() + k[1:]
    # try:
    #     title += ' population={:.1f}M'.format(d_country2pop[args.title])
    # except KeyError:
    #     pass
    title += ', {}'.format(args.dateToday)
    title += '\nCases today={}, Deaths today={}'.format(
        df['cases'].values[-1],
        df['deaths'].values[-1],
        )
    if tFit is not None:
        title += '\nCalculated cumulated {}={:d}, midpoint={:d}, steepness={:.2f}'.format(
            k, int(popt[0]), int(popt[2]), popt[1])
    title += '\nCurrent day={}'.format(len(df))
    title += ', Day of first case={}'.format(dayFirstCase)
    title += ', Total confirmed cases={}'.format(max(df['cases'].values.cumsum()))
    title += ', Total confirmed deaths={}'.format(int(max(df['deaths'].values.cumsum())))
    plt.title(title, fontsize='x-small')
    plt.legend()
    # plt.yscale('log')
    path = 'COVID19_sigmoid_{}_{}_{}.png'.format(k, args.affix, args.dateToday)
    print(path)
    plt.savefig(path, dpi=80)
    path = 'COVID19_sigmoid_{}_{}.png'.format(k, args.affix)
    print(path)
    plt.savefig(path, dpi=80)
    print(path)
    plt.clf()

    if k == 'cases':

        popSize = args.d_country2pop[args.title]
        s = '<tr>'
        s += '<td>{}</td>'.format(args.title)
        s += '<td>{:.1f}</td>'.format(popSize)
        try:
            s += '<td>{:.2f}</td>'.format(fitSteep)
            print(s)
        except UnboundLocalError:
            s += '<td>N/A</td>'
            print(s)
            pass
        s += '<td>{}</td>'.format(df['cases'].values.sum())
        s += '<td>{}</td>'.format(int(df['deaths'].values.sum()))
        s += '<td>{:.1f}</td>'.format(100 * df['deaths'].values.sum() / df['cases'].values.sum())
        s += '<td>{}</td>'.format(df['cases'].values[-1])
        s += '<td>{}</td>'.format(df['deaths'].values[-1])
        try:
            s += '<td>{:.1f}</td>'.format(df['cases'].values.sum() / popSize)
            s += '<td>{:.1f}</td>'.format(df['deaths'].values.sum() / popSize)
        except UnboundLocalError:
            pass
        s += '</tr>'
        with open('table{}.txt'.format(args.affix), 'w') as f:
            print(s, file=f)

    return


def doScatterPlots(args, df0):

    for region in args.d_region2countries.keys():
        for k in ('cases', 'deaths'):
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
            plt.ylabel(k[0].upper() + k[1:])    
            plt.legend(prop={'size': 6})
            plt.title(region + '\n' + k[0].upper() + k[1:])
            # plt.label()
            path = 'scatter_{}_{}.png'.format(region, k)
            plt.savefig(path, dpi=80)
            plt.clf()

    return



def doLinePlots(args, df0):

    for region in args.d_region2countries.keys():

        for perCapita in (True, False,):
            # for language in ('en', 'es',):
            for language in ('en',):
                for k, limSum in (('cases', 20000), ('deaths', 500)):
                    print('line', region, perCapita, language, k)
                    k_loc = {
                        'en': {
                            'cases': 'cases',
                            'deaths': 'deaths',
                            },
                        'es': {
                            'cases': 'casos',
                            'deaths': 'muertos',
                            },
                            }[language][k]
                    l = []
                    # for country in df0['countriesAndTerritories'].unique():
                    for country in args.d_region2countries[region]:
                        if df0[df0['countriesAndTerritories'].isin([country])][k].sum() == 0:
                            continue
                        if perCapita is True:
                            value = operator.truediv(
                                10**6 * df0[df0['countriesAndTerritories'].isin([country])][k].sum(),
                                df0[df0['countriesAndTerritories'].isin([country])]['popData2018'].unique(),
                                )[0]
                            if np.isnan(value):
                                continue
                            l.append((value, country))
                        else:
                            l.append((
                                df0[df0['countriesAndTerritories'].isin([country])][k].sum(),
                                country,
                                ))
                    for t in reversed(sorted(l)):
                        country = t[1]
                        df = df0[df0['countriesAndTerritories'].isin([country])].sort_values(by='dateRep', ascending=True)
                        if k == 'deaths' and 'South' in country:
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
                        print(country, df[k].sum())
                        # print(country, df[k].sum())
                        if perCapita is False:
                            lim = {'cases': 100, 'deaths': 10}[k]
                            y = df[k].cumsum()[df[k].cumsum() > lim].values
                        else:
                            lim = 1
                            # s = 10**6 * df[k].cumsum()[df[k].cumsum() > 100].values / df['popData2018'].unique()[0]
                            # y = s
                            s = 10**6 * df[k].cumsum() / df['popData2018'].unique()
                            y = s[s > lim]
                            y = df[k].cumsum()[df[k].cumsum() > 100].values / df['popData2018'].unique()
                        if df[k].sum() < lim:
                            continue
                        if len(y) == 0:
                            continue
                        x = list(range(len(y)))
                        if language == 'es':
                            country = {
                                'United Kingdom': 'Reino Unido',
                                'United States of America': 'EE.UU.',
                                'China': 'China',
                                'Italy': 'Italia',
                                'Spain': 'España',
                                'Germany': 'Alemania',
                                'France': 'Francia',
                                'Switzerland': 'Suiza',
                                'Japan': 'Japon',
                                'Singapore': 'Singapur',
                                'South Korea': 'Corea del Sur',
                                'Netherlands': 'Países Bajos',
                                # 'Austria': 'Países Bajos',
                                'Taiwan': 'Taiwán',
                                'Belgium': 'Belgica',
                                'Turkey': 'Turquía',
                                }[country]
                        country = country.replace('United States of America', 'US')
                        country = country.replace('United Kingdom', 'UK')
                        plt.semilogy(
                            x, y,
                            label='{} ({:d})'.format(
                                country.replace('_', ' '),
                                int(max(y)),
                                ),
                            linewidth=2,
                            )
                    plt.legend(prop={'size': 6})
                    if perCapita is True:
                        textPerCapita = ' {} 1 million capita'.format({
                            'en': 'per', 'es': 'por'}[language])
                    else:
                        textPerCapita = ''
                    if lim == 1:
                        kSingPlur = k_loc.lower()[:-1]
                    else:
                        kSingPlur = k_loc.lower()
                    if language == 'en':
                        plt.xlabel('Days since {} confirmed {}{}'.format(lim, kSingPlur, textPerCapita))
                        plt.ylabel('Cumulated confirmed {}{}'.format(k_loc.lower(), textPerCapita))
                    else:
                        plt.xlabel('Dias desde {} {} confirmados {}'.format(lim, kSingPlur, textPerCapita))
                        plt.ylabel('{} confirmados acumulados{}'.format(k_loc[0].upper(), k_loc[1:]), textPerCapita)
                    text = {
                        'en': 'after first day with more than ',
                        'es': 'desde el primer dia con ',
                        }[language]
                    if lim == 1:
                        textLim = '{}{}'.format(k_loc.lower()[:-1], textPerCapita)
                    else:
                        textLim = '{}{}'.format(k_loc.lower(), textPerCapita)
                    keyUpperCase = '{}{}'.format(k_loc[0].upper(), k_loc[1:])
                    plt.title('{}\n{}{} {} {} {}'.format(
                        region, keyUpperCase, textPerCapita, text, lim, textLim), fontsize='small')
                    path = 'days100_{}_{}_perCapita{}_{}.png'.format(k_loc, language, perCapita, region)
                    plt.savefig(path, dpi=80)
                    plt.clf()

    return


def fit(args, df, xConfCasesCumYesterday, yConfCasesCumYesterday):

    # Seed values for regression. Heuristic approach.
    if 0 in yConfCasesCumYesterday:
        guessMidpoint = min(60, 40 + list(yConfCasesCumYesterday[::-1]).index(0))
    else:
        guessMidpoint = 40
    # Cases less than 5 percent of ConfCasesCum
    if df['cases'].values[-2] / yConfCasesCumYesterday[-1] < 0.05:
        guessCasesMax = 1.5 * max(yConfCasesCumYesterday)
    else:
        guessCasesMax = 10 * max(yConfCasesCumYesterday)
    print('guessMidpoint', guessMidpoint)
    print('guessCasesMax', guessCasesMax)
    # guessCasesMax = 100000
    # guessMidpoint = 76
    p0 = [guessCasesMax, 0.5, guessMidpoint]
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
    try:
        popt, pcov = curve_fit(
            logistic,
            xConfCasesCumYesterday,
            yConfCasesCumYesterday,
            p0=p0,
            )
        perr = np.sqrt(np.diag(pcov))
        print('maximum', popt[0])
        print('midpoint', popt[2] - 3 * perr[2], popt[2] + 3 * perr[2])
        print('steepness', popt[1])
    except RuntimeError:
        print('Fitting failed')
        print('\n'.join('{}\t{}\t{}'.format(*t) for t in zip(
            xConfCasesCumYesterday, yConfCasesCumYesterday,
            map(int, df['deaths'].values.cumsum()),
            )))
        return
    except:
        print('Exception')
        print('\n'.join('{}\t{}\t{}'.format(*t) for t in zip(
            xConfCasesCumYesterday, yConfCasesCumYesterday,
            df['deaths'].values.cumsum(),
            )))
        return

    # Heuristic checks of whether curve fitting makes sense.
    booleans = (
        # Calculated cases plus 10 percent should be greated than actual cases today.
        1.1 * popt[0] < df['cases'].values.sum(),
        # Steepness should not exceed 1.
        popt[1] > 1,
        # Error greater than half of calculated or actual maximum.
        3 * perr[0] > .5 * popt[0],
        perr[0] > df['cases'].values.sum(),
        )

    if any(booleans):
        print('\n'.join('{}\t{}\t{}'.format(*t) for t in zip(
            df.index.strftime('%Y-%m-%d').values,
            df['cases'].values.cumsum(),
            df['deaths'].values.cumsum(),
            )))
        print('Fitting yielded unreliable results for {}.'.format(args.title))
        print('popt', popt)
        print('perr', perr)
        print('ConfCasesSum', df['cases'].values.sum())
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


def parseURL(url):

    basename = os.path.basename(url)
    if not os.path.isfile(basename):
        r = requests.get(url)
        if r.status_code != 200:
            print(url)
            exit()
        # with open(basename, 'w') as f:
            # f.write(r.content)
        with open(basename, 'w') as f:
            f.write(r.text)
    # df = pd.read_excel(basename)
    df = pd.read_csv(basename)

    return df


def parseArgs():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--countries', nargs='*',
        help='e.g. Denmark,Sweden,Norway,Finland,Iceland',
        )
    parser.add_argument(
        '--region',
        choices=(
            'Europe',
            'EU',
            'Scandinavia',
            'Nordic',
            'EuropeMediterranean',
            'EuropeSouth',
            'EuropeEast',
            'EuropeWest',
            'EuropeNorth',
            'Oceania',
            'LatinAmerica',
            'LatinAmericaExVenezuela',
            'AsiaSouthEast',
            'AsiaCentral',
            'AsiaEast',
            'AsiaSouth',
            'AsiaWestern',
            'Oceania',
            'Americas',
            'AmericaSouth',
            'AmericaSouthExVenezuela',
            'AmericaNorth',
            'Africa',
            'AsiaWesternExIran',
            'AsiaEastExChina',
            'AsiaExChina',
            'Topol',
            'EuropeNW',
            ),
        )
    parser.add_argument(
        '--dateToday', default=datetime.today().strftime('%Y-%m-%d'),
        help='Date in ISO 8601 format YYYY-MM-DD',
        required=False,
        )
    args = parser.parse_args()

    d_region2countries = {
        'EU': [
            'Austria',
            'Belgium',
            'Bulgaria',
            'Croatia',
            'Cyprus',
            'Czech Republic',
            'Denmark',
            'Estonia',
            'Finland',
            'France',
            'Germany',
            'Greece',
            'Hungary',
            'Ireland',
            'Italy',
            'Latvia',
            'Lithuania',
            'Luxembourg',
            'Malta',
            'Netherlands',
            'Poland',
            'Portugal',
            'Romania',
            'Slovakia',
            'Slovenia',
            'Spain',
            'Sweden',
            ],
        'AsiaSouthEast': (
            'Indonesia',
            'Thailand',
            'Philippines',
            'Malaysia',
            'Singapore',
            'Vietnam',
            'Cambodia',
            'Brunei Darussalam',
            'Myanmar',
            ),
        'AsiaCentral': (
            'Afghanistan',
            'Kazakhstan',
            'Uzbekistan',
            'Kyrgyzstan',
            ),
        'AsiaEast': (
            'China',
            'Japan',
            'Mongolia',
            'South Korea',
            # 'Hong Kong',
            'Taiwan',
            ),
        'AsiaSouth': (
            'India',
            'Bangladesh',
            'Afghanistan',
            'Maldives',
            'Nepal',
            'Bhutan',
            'Sri Lanka',
            'Pakistan',
            ),
        'AsiaWestern': (
            'Armenia',
            'Azerbaijan',
            'Bahrain',
            'Egypt',
            'Qatar',
            'Kuwait',
            'Oman',
            'United Arab Emirates',
            'Saudi Arabia',
            'Israel',
            'Iran',
            'Iraq',
            'Georgia',
            'Turkey',
            'Lebanon',
            'Jordan',
            'Palestine',
            ),
        'AmericaSouth': (
            'Brazil',
            'Colombia',
            'Argentina',
            'Peru',
            'Venezuela',
            'Chile',
            'Ecuador',
            'Bolivia',
            'Paraguay',
            'Uruguay',
            'Guyana',
            'Suriname',
            'El Salvador',
            ),
        'AmericaNorth': (
            'United States of America',
            'Mexico',
            'Canada',
            'Guatemala',
            'Honduras',
            'Trinidad and Tobago',
            'Costa Rica',
            'Dominican Republic',
            'Panama',
            'Saint Vincent and the Grenadines',
            'Antigua and Barbuda',
            'Saint Lucia',
            'Jamaica',
            'Cuba',
            'Bahamas',
            'Barbados',
            'Bermuda',
            'Cayman Islands',
            'Netherlands Antilles',
            'Haiti',
            'Nicaragua',
            ),
        'Africa': (
            'Algeria',
            'Burkina Faso',
            'Democratic Republic of the Congo',
            'Cote dIvoir',
            'Cameroon',
            'Egypt',
            'Gabon',
            'Ethiopia',
            'South Africa',
            'Senegal',
            'Ghana',
            'Nigeria',
            'Cote dIvoire',
            'Guinea',
            'Kenya',
            'Togo',
            'Morocco',
            'Sudan',
            'Tunisia',
            'Namibia',
            'Swaziland',
            'Equatorial Guinea',
            'Mauritania',
            'Seychelles',
            'Rwanda',
            'Central African Republic',
            'Congo',
            'United Republic of Tanzania',
            'Somalia',
            'Benin',
            'Eswatini',
            'Liberia',
            'Gambia',
            'Djibouti',
            'Zambia',
            'Chad',
            'Mauritius',
            'Zimbabwe',
            'Niger',
            'Madagascar',
            ),
        'Oceania': (
            'Australia',
            # 'Papua New Guinea',
            'New Zealand',
            # 'Fiji',
            'French Polynesia',
            'Guam',
            'Fiji',
            'Papua New Guinea',
            ),
        'Scandinavia': ('Denmark', 'Sweden', 'Norway'),
        'Nordic': ('Denmark', 'Sweden', 'Norway', 'Iceland', 'Finland', 'Greenland', 'Faroe Islands'),
        'EuropeMediterranean': (
            'Greece', 'Spain', 'France', 'Monaco', 'Italy', 'Malta',
            'Slovenia', 'Croatia', 'Bosnia and Herzegovina',
            'Montenegro', 'Albania', 'Turkey',
            ),
        }
    d_region2countries['EuropeWest'] = [
        'Austria',
        'Belgium',
        'Czech Republic',
        'France',
        'Germany',
        'Ireland',
        'Liechtenstein',
        'Luxembourg',
        'Monaco',
        'Netherlands',
        'Switzerland',
        'United Kingdom',
        ]
    d_region2countries['EuropeEast'] = [
        'Estonia',
        'Latvia',
        'Lithuania',
        'Armenia',
        'Azerbaijan',
        'Georgia',
        ]
    d_region2countries['Europe'] = list(d_region2countries['EU'])
    d_region2countries['Europe'].extend([
        'Albania',
        'Bosnia and Herzegovina',
        'Belarus',
        'Moldova',
        'Russia',
        'Norway',
        'Switzerland',
        'Ukraine',
        'United Kingdom',
        'Monaco',
        'Iceland',
        'Andorra',
        'Serbia',
        'Holy See',
        'North Macedonia',
        'San Marino',
        'Liechtenstein',
        'Kosovo',
        'Gibraltar',
        'Greenland',
        'Jersey',
        'Faroe Islands',
        'Guernsey',
        'Isle of Man',
        ])
    x = ['Andorra', 'Armenia', 'Australia', 'Azerbaijan', 'Bahrain', 'Belarus', 'Bosnia and Herzegovina', 'Brunei Darussalam', 'Burkina Faso', 'Cambodia', 'Cameroon', 'Cases on an international conveyance Japan', 'Chile', 'Colombia', 'Costa Rica', 'Cote dIvoire', 'Cuba', 'Democratic Republic of the Congo', 'Dominican Republic', 'Ecuador', 'Egypt', 'Ethiopia', 'Gabon', 'Georgia', 'Ghana', 'Guinea', 'Guyana', 'Holy See', 'Honduras', 'Iceland', 'Iran', 'Iraq', 'Israel', 'Jordan', 'Kenya', 'Kuwait', 'Lebanon', 'Liechtenstein', 'Malaysia', 'Maldives', 'Mexico', 'Moldova', 'Monaco', 'Mongolia', 'Morocco', 'Nepal', 'New Zealand', 'Nigeria',
    'North Macedonia', 'Oman', 'Pakistan', 'Palestine', 'Panama', 'Paraguay', 'Philippines', 'Qatar', 'Russia', 'Saint Vincent and the Grenadines', 'San Marino', 'Saudi Arabia', 'Senegal', 'Serbia', 'South Africa', 'Sri Lanka', 'Sudan', 'Switzerland', 'Taiwan', 'Thailand', 'Togo', 'Trinidad and Tobago', 'Tunisia', 'Turkey', 'Ukraine', 'United Arab Emirates', 'United Kingdom', 'United kingdom', 'Vietnam', 'switzerland']
    d_region2countries['Americas'] = d_region2countries['AmericaSouth'] + d_region2countries['AmericaNorth']
    d_region2countries['Asia'] = d_region2countries['AsiaSouthEast'] + d_region2countries['AsiaCentral'] + d_region2countries['AsiaEast'] + d_region2countries['AsiaSouth'] + d_region2countries['AsiaWestern']
    d_region2countries['AsiaExChina'] = set(d_region2countries['Asia']) - set(['China'])
    d_region2countries['AsiaEastExChina'] = set(d_region2countries['AsiaEast']) - set(['China'])
    d_region2countries['AsiaWesternExIran'] = set(d_region2countries['AsiaWestern']) - set(['Iran'])
    d_region2countries['LatinAmerica'] = set(d_region2countries['AmericaNorth'] + d_region2countries['AmericaSouth']) - set(['United States of America', 'Canada'])
    d_region2countries['LatinAmericaExVenezuela'] = set(d_region2countries['LatinAmerica']) - set(['Venezuela'])
    d_region2countries['AmericaSouthExVenezuela'] = set(d_region2countries['AmericaSouth']) - set(['Venezuela'])
    d_region2countries['Topol'] = [
        'United States of America', 'Senegal', 'Israel', 'Austria',
        'Malaysia', 'Greece', 'Estonia',
        'Taiwan',
        'New_Zealand',
        'Australia',
        'South_Korea',
        'Uruguay',
        ]
    d_region2countries['EuropeNW'] = [
        'Denmark',
        'Sweden',
        'Norway',
        'Finland',
        'Iceland',

        'United_Kingdom',
        'Germany',
        'Netherlands',
        'Switzerland',
        'Poland',
        'Belgium',
        'Austria',
        'Ireland',

        # 'France',

        # 'Italy',
        # 'Spain',
        ]

    d_region2countries['World'] = [
        'United States of America',
        # 'China',  # fake numbers?
        # 'Iran',  # fake numbers?
        'Italy', 'Spain', 'Germany', 'France',
        'United_Kingdom',
        'South_Korea',
        # 'Japan',
        # 'Singapore',
        'Taiwan',
        'New Zealand',
        'Uruguay',
        'Senegal',
        'Vietnam',
        'Greece',
        'Australia',
        'Denmark',
        'Austria',
        'Estonia',
        'Iceland',
        'Israel',
        'South Africa',
        'Senegal',
        'New Zealand',
        'Norway',
        'Malaysia',
        ]

    d_region2countries['WorldAll'] = set()
    for region in d_region2countries.keys():
        d_region2countries['WorldAll'] |= set(d_region2countries[region])

    print(args)

    # # temporary to check for new countries. tmp
    # domain = 'https://www.ecdc.europa.eu'
    # basename = 'COVID-19-geographic-disbtribution-worldwide-{}.xlsx'.format(
    #     args.dateToday)
    # url = '{}/sites/default/files/documents/{}'.format(domain, basename)
    # df = parseURL(url)
    # countries = [_.replace('_',' ') for _ in df['countriesAndTerritories'].unique()]
    # for k in d_region2countries.keys():
    #     for t in d_region2countries[k]:
    #         try:
    #             countries.remove(t)
    #         except:
    #             continue
    # l = list(sorted(countries))
    # assert l == [
    # 'CANADA',
    # 'Cases on an international conveyance Japan',], l
    # # assert len(l) == 1

    if args.countries is not None:
        args.countries = '_'.join(args.countries).split(',')
        args.title = ','.join((_.replace('_', ' ') for _ in args.countries))
        args.affix = ''.join(args.countries).replace(' ', '_')
    elif args.region is not None:
        args.countries = d_region2countries[args.region]
        args.title = args.region
        args.affix = args.region
    else:
        args.countries = [_.replace('_',' ') for _ in df['countriesAndTerritories'].unique()]
        args.title = 'World'
        args.affix = 'World'

    args.d_region2countries = d_region2countries

    d_country2pop = {
        'China': 1433.7,
        'Italy': 60.5,
        'South Korea': 51.2,
        'Spain': 46.7,
        'France': 65.1,
        'US': 329.0,
        'Japan': 126.9,
        'Singapore': 5.8,
        'Taiwan': 23.8,
        'EU': 512.6,
        'Sweden': 10.3,
        'Canada': 37.59,
        'India': 1339,
        'Brazil': 209.3,
        'Germany': 82.8,
        'Australia': 82.8,
        'Denmark': 5.6,
        'AsiaEastExChina': 1700-1386,
        'United Kingdom': 66.44,
        'United States of America': 327.2,
    }

    args.d_country2pop = d_country2pop

    return args


if __name__ == '__main__':
    main()
