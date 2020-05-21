import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import math
from adjustText import adjust_text
import numpy as np
from datetime import date

eu27 = [
    'AUT',
    'BEL',
    'BGR',
    'HRV',
    'CYP',
    'CZE',
    'DNK',
    'EST',
    'FIN',
    'FRA',
    'DEU',
    'GRC',
    'HUN',
    'IRL',
    'ITA',
    'LVA',
    'LTU',
    'LUX',
    'MLT',
    'NLD',
    'POL',
    'PRT',
    'ROU',
    'SVK',
    'SVN',
    'ESP',
    'SWE',
    ]

europe = eu27 + [
    'ISL',
    'NOR',
    'GBR',
    'CHE',
    'FRO',
    ]

g7 = [
    'CAN',
    'JPN',
    'USA',
    'DEU',
    'FRA',
    'ITA',
    'GBR',
    ]

latin_america = [
    'PER',
    'COL',
    'CHL',
    'ARG',
    'BOL',
    'URY',
    'BRA',
    'MEX',
    'ECU',
    'GTM', 'GCA',
    'CUB',
    'HTI',
    'DOM',
    'SLV',
    'PAN',
    'CRI',
    ]

asia = [
'AFG',
'ARM',
'AZE',
'BHR',
'BGD',
'BTN',
'BRN',
'KHM',
'CHN',
'CXR',
'CCK',
'IOT',
'GEO',
'HKG',
'IND',
'IDN',
'IRN',
'IRQ',
'ISR',
'JPN',
'JOR',
'KAZ',
'KWT',
'KGZ',
'LAO',
'LBN',
'MAC',
'MYS',
'MDV',
'MNG',
'MMR',
'NPL',
'PRK',
'OMN',
'PAK',
'PSE',
'PHL',
'QAT',
'SAU',
'SGP',
'KOR',
'LKA',
'SYR',
'TWN',
'TJK',
'THA',
'TUR',
'TKM',
'ARE',
'UZB',
'VNM',
'YEM',
]

americas = latin_america + ['USA', 'CAN']

africa = [
'MAR',
'DZA',
'ZAF',
'TUN',
'NGA',
'ETH',
'MDG',
'MUS',
'LBY',
'COD',
'KEN',
'SYC',
'UGA',
'GHA',
'CPV',
'SDN',
'MLI',
'TZA',
'SEN',
'SOM',
'CIV',
'ZWE',
'BFA',
'CMR',
'RWA',
'AGO',
'REU',
'MOZ',
'ERI',
'NER',
'TCD',
'GIN',
'MRT',
'NAM',
'SWZ',
'TGO',
'DJI',
'LBR',
'SLE',
'BEN',
'GAB',
'GMB',
'ZMB',
'MWI',
'BWA',
'BDI',
'LSO',
'SSD',
'COG',
'COM',
'GNQ',
]

oceania = [
'AUS',
'NZL',
'FJI',
'GUM',
'WSM',
'NCL',
'ASM',
'PLW',
'PNG',
'PYF',
'VUT',
'NRU',
'TUV',
'TON',
'COK',
'KIR',
'SLB',
'MHL',
'PCN',
'FSM',
'NIU',
'MNP',
'NFK',
'WLF',
]

d_regions = {
    'Americas': americas,
    'Europe': europe,
    'Asia': asia,
    'Africa': africa,
    'Oceania': oceania,
    'G7': g7,
    'EU': eu27,
    'Nordic': ['DNK', 'SWE', 'NOR', 'FRO', 'ISL', 'FIN']
}

def main():

    url = 'https://opendata.ecdc.europa.eu/covid19/casedistribution/csv'
    # df_ecdc = pd.read_html(url)
    df_ecdc = pd.read_csv('csv')

    df_owid = pd.read_csv('owid.csv')

    for region in (
        'Americas',
        'Nordic',
        'Europe',
        'Asia',
        'Oceania',
        'Africa',
        'G7',
        'EU',
        ):
        countries = d_regions[region]

        x = []
        y = []
        colors = []
        l_total_tests_per_thousand = []
        sizes = deathsCumulative = []
        labels = []
        z = []

        # print(df_ecdc)
        # print(df_owid)
        # print(df_ecdc['countryterritoryCode'].unique())
        # exit()
        for country in df_ecdc['countryterritoryCode'].unique():
            if country not in df_owid['iso_code'].unique():
                continue
            # if country not in europe + g7:
            #     continue
            # if country not in americas:
            #     continue
            if country not in countries:
                continue
            cases = df_ecdc[df_ecdc['countryterritoryCode'] == country]['cases']
            deaths = df_ecdc[df_ecdc['countryterritoryCode'] == country]['deaths']
            x2 = cases.head(7).sum()
            x1 = cases.head(14).tail(7).sum()
            # if country == 'LTU':
            #     continue  # negative values???
            #     print(cases.to_string())
            #     exit()
            # if country in ('CYP',):
            #     continue  # no testing?
            y2 = deaths.head(7).sum()
            y1 = deaths.head(14).tail(7).sum()
            # if x1 == 0 or y1 == 0:
            #     continue

            total_tests_per_thousand = df_owid[df_owid['iso_code'] == country]['total_tests_per_thousand']
            if total_tests_per_thousand.max() == 0:
                continue

            label = df_owid[df_owid['iso_code'] == country]['location'].iloc[0]

            deathsTotal = df_ecdc[df_ecdc['countryterritoryCode'] == country]['deaths'].sum()

            # changeCasesWeekly = 100 * (x2 - x1) / x1
            # changeDeathsWeekly = 100 * (y2 - y1) / y1

            x1 = cases.sum() - cases.head(7).sum()
            x2 = cases.sum()
            changeCasesWeekly = 100 * (x2 - x1) / x1
            # Skip outliers.
            if changeCasesWeekly > 100:
                continue

            y1 = deaths.sum() - deaths.head(7).sum()
            y2 = deaths.sum()
            if y1 == 0:  # Faroe Islands have zero deaths
                changeDeathsWeekly = 0
            else:
                changeDeathsWeekly = 100 * (y2 - y1) / y1
            # Skip outliers.
            if changeDeathsWeekly > 100:
                continue

            z2 = total_tests_per_thousand.max()
            # Skip if no testing.
            if z2 == 0 or np.isnan(z2):
                if country in ('FRO',):
                    z.append(0)
                else:
                    continue
            else:
                for i in range(7, 21):
                    z1 = total_tests_per_thousand.iloc[-i]
                    if z1 == z2:
                        continue
                    if not np.isnan(z1):
                        z.append(100 * (z2 - z1) / z1)
                        break
                else:
                    z.append(0)
                    # print(country, z2)
                    # print(total_tests_per_thousand.to_string())
                    # exit()

            # z.append(100 * (z2 - z1) / z1)

            l_total_tests_per_thousand.append(total_tests_per_thousand.max())

            x.append(changeCasesWeekly)
            y.append(changeDeathsWeekly)
            labels.append(label)
            # sizes.append(5 * math.sqrt(deaths))
            size = 10 * deathsTotal ** (1/3)
            sizes.append(size)
            # print(country, changeCasesWeekly, changeDeathsWeekly, size)

        # for _ in l_total_tests_per_thousand:
        #     colors.append(_ / max(l_total_tests_per_thousand))

        colors = z

        fig, ax = plt.subplots()

        fig.set_size_inches(16 / 2, 9 / 2)

        for t in reversed(sorted(zip(sizes, x, y, colors, labels))):
            sizes.append(t[0])
            x.append(t[1])
            y.append(t[2])
            colors.append(t[3])
            labels.append(t[4])
        sizes = sizes[len(sizes)//2:]
        x = x[len(x)//2::]
        y = y[len(y)//2::]
        colors = colors[len(colors)//2::]
        labels = labels[len(labels)//2::]

        paths = ax.scatter(
            x, y,
            c = colors,
            s = sizes,
            marker = 'o',
            # label = labels,
            edgecolor = 'black',
            linewidth = 0.1,
            # font = {
            #     # 'family' : 'normal',
            #     # 'weight' : 'bold',
            #     'size'   : 'small',
            #     },
            alpha = 0.5,
            cmap = 'viridis',
            # vmin = min(l_total_tests_per_thousand),
            # vmax = max(l_total_tests_per_thousand),
            vmin = 0,
            vmax = max(colors),
            # vmax = 25,
            )

        # ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        # ax.axvline(0, color='black', linewidth=0.5, linestyle='--')

        cbar = fig.colorbar(paths)
        # cbar.set_label('Total tests per thousand')
        cbar.set_label('Weekly change in total tests (%)')

        texts = []
        for xi, yi, label in zip(x, y, labels):
            texts.append(ax.text(
                xi, yi, label,
                size='xx-small',
        ##        ha='left', va='bottom',
                ha='center', va='center',
                color='white',
                path_effects=[
                    path_effects.Stroke(linewidth=1, foreground='black'),
                    path_effects.Normal(),
                    ],
                ))

        # adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black'))
        adjust_text(texts)

        # legend_elements = []
        # print(colors)
        # print(max(colors))
        # print(l_total_tests_per_thousand)
        # print(max(l_total_tests_per_thousand))
        # print(min(l_total_tests_per_thousand))
        # exit()
        # for total_tests_per_thousand in [0, ]:
        #     label = forwardPE
        #     if forwardPE == 30:
        #         label = '30+'
        #     legend_elements.append(
        #         Line2D(
        #             [0], [0], marker='o',
        #             color = 'white',  # don't show line
        #             label=label,
        #             markerfacecolor=color_forwardPE(forwardPE),
        #             markersize=10,
        #             ))
        # legend_color = ax.legend(
        #     handles=legend_elements,
        #     loc='upper right',
        #     title=args.color,
        #     # bbox_to_anchor=(0.75, 1),
        #     bbox_to_anchor=(0.95, 1),
        #     )

        # # produce a legend with a cross section of sizes from the scatter
        # handles, labels = scatter.legend_elements(
        #     prop="sizes", alpha=0.5,
        #     func=lambda s: s / 3.01,
        #     )
        # labels[-1] = '5.0+'
        # labels = labels[0::2]
        # handles = handles[0::2]
        # legend_size = ax.legend(
        #     handles, labels, loc="upper right",
        #     title=args.size,
        #     # func=lambda s: np.sqrt(s) / 4,
        #     )
        # ax.add_artist(legend_size)

        ax.set_xlabel('Weekly change in total cases (%)')
        ax.set_ylabel('Weekly change in total fatalities (%)')
        ax.set_title(region)

        # ax.set_xlim(0, 75)  # El Salvador
        # ax.set_ylim(0, 65)  # Mexico
        # ax.set_xlim(0, 100)  # Ghana
        # ax.set_ylim(0, 120)  # Senegal
        # ax.set_xlim(0, 5 + 5 * max((25, max(x), max(y))) // 5)
        # ax.set_ylim(0, 5 + 5 * max((25, max(x), max(y))) // 5)
        ax.set_xlim(0, min(100, 2 + 2 * max((20, max(x))) // 2))
        ax.set_ylim(0, min(100, 2 + 2 * max((20, max(y))) // 2))
        # ax.set_xlim(0, 25)  # Canada
        # ax.set_ylim(0, 30)  # Canada

        path = 'plot_bubble_{}.png'.format(region)
        fig.savefig(path, dpi=200)
        print(path)
        fig.savefig('plot_bubble_{}_{}.png'.format(region, date.today().isoformat()), dpi=200)

    return

if __name__ == '__main__':
    main()
