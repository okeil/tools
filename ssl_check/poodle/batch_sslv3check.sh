#!/bin/bash
targets=$1
port=${2-443}
scanDate=`date +%Y%m%d`
result=$scanDate-$targets-sslv3_check.csv

echo "host,port,sni_result,no_sni_result" > $result
for X in $(cat $targets);do
	echo $X,$port,$(./sslv3_check.sh $X $port 1),$(./sslv3_check.sh $X $port 0) >> $result
done
