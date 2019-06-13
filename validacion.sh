#!/bin/bash

#AIC_VALUE=$(awk -F,\" '{ print $13 }' resultados.csv  | awk -F\" '{print $1}' | sed -n 2p)
#sed -i '1 d' r_AIC.txt

if [ $(cat r_AIC.txt) = "Yes" ]
then
echo "Modelo Exitoso"
else
echo "Modelo no Exitoso"
exit 1
fi
