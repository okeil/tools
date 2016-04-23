#!/bin/bash
cd /root/pbnj_scanning
./runNMAPscans.sh 1> /dev/null 2> ./scan_errors.log
./runPBNJimport.sh 1> /dev/null 2> ./import_errors.log
if [ `/usr/local/bin/outputpbnj --query daily_all | wc -l` -gt 0 ]; then
	./getLatest.sh
fi
if [ $(wc -l < scan_errors.log) -gt 0 ] || [ $(wc -l < scan_errors.log) -gt 0 ]; then
	echo "# SCAN ERRORS #"
	cat ./scan_errors.log
	echo "# IMPORT ERRORS #"
	cat ./import_errors.log
fi
