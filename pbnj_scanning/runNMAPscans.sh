#!/bin/bash
logfile=nmap_scan.log
echo "`date +%d-%m-%Y-%T` --- STARTING scan ---" > $logfile
scanDate=`date +%d-%m-%Y-%T`

/bin/mv output/*.xml output/archive/

for X in lists/*.ipv4.list; do
	scanName=$(echo $X | sed -n 's/^.*\/\(.*\),*/\1/p') 
	echo "          `date +%d-%m-%Y-%T` - $scanName started" >> $logfile
	nmap -T4 -open -oX output/$scanDate.$scanName.xml -iL $X
	echo "		`date +%d-%m-%Y-%T` - $X completed" >> $logfile
done
for X in lists/*.ipv6.list; do
        echo "          `date +%d-%m-%Y-%T` - $X started" >> $logfile
        nmap -6 -T4 -open -oX output/$scanDate.$scanName.xml -iL $X
        echo "          `date +%d-%m-%Y-%T` - $X completed" >> $logfile
done
echo "`date +%d-%m-%Y-%T` --- COMPLETED scan ---" >> $logfile
