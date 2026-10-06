if [ -s COVID19_sigmoid_cases_EU.png ]; then exit; fi

rm days100*.png
rm csv

wget https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv
mv owid-covid-data.csv owid.csv

for region in \
EU \
Nordic \
EuropeMediterranen \
Europe \
Americas \
LatinAmericaExVenezuela \
AmericaSouthExVenezuela \
AmericaSouth \
AmericaNorth \
LatinAmerica \
AsiaSouthEast \
AsiaCentral \
AsiaSouth \
Africa \
AsiaWesternExIran \
AsiaEastExChina \
AsiaExChina \
Oceania \
; do
 python3 plot_series.py --region $region
# if [ ! -s COVID19_sigmoid_cases_EU.png ]; then exit; fi
done

while read country; do
 echo; echo $country
 python3 plot_series.py --countries $country
done < countries.txt

mv *2020*.png archive

mv table*txt tables ; cat tables/*.txt | grep -v Asia

(cd regional_maps/europe && python3 plot_choropleth_europe.py && mv europe_*.png europe.gif ../..)

python3 plot_choropleth.py

python3 plot_bubble.py

mv *[0-9].png archive

python3 upload.py

mv *.png archive
