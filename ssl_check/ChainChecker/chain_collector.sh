#!/bin/bash

# chain_collector.sh [domain] [port]
# output to stdout

if [ $# -ne 2 ]; then
	echo "USAGE: chain_collector.sh [domain] [port]"
	exit 1
fi

SERVER=$1:$2
TFILE="/tmp/$(basename $0).$$.tmp"
OUTPUT_DIR=$1_$2
mkdir $OUTPUT_DIR

openssl s_client -showcerts -servername $1 -connect $SERVER 2>/dev/null </dev/null > $TFILE
awk 'BEGIN {c=0;} /BEGIN CERT/{c++} { print > "tmpcert." c ".pem"}' < $TFILE
i=1
for X in tmpcert.*.pem; do
	if openssl x509 -noout -in $X 2>/dev/null ; then 
		echo "#############################"
                cn=$(openssl x509 -noout -subject -in $X | sed -e 's#.*CN=\(\)#\1#')
		echo CN: $cn
		cp $X $OUTPUT_DIR/${cn// /_}.$((i-1)).pem 
		cert_expiry_date=$(openssl x509 -noout -enddate -in $X \
				| awk -F= ' /notAfter/ { printf("%s\n",$NF); } ')
		seconds_until_expiry=$(echo "$(date --date="$cert_expiry_date" +%s) - $(date +%s)" |bc)
        	days_until_expiry=$(echo "$seconds_until_expiry/(60*60*24)" |bc)
		echo Days until expiry: $days_until_expiry
		echo $(openssl x509 -noout -text -in $X | grep -m1 "Signature Algorithm:" | head)
		echo $(openssl x509 -noout -issuer -in $X)
		if [ -a tmpcert.$i.pem ]; then
			echo Parent: $(openssl x509 -noout -subject -in tmpcert.$i.pem | sed -e 's#.*CN=\(\)#\1#')
			echo Parent Valid? $(openssl verify -verbose -CAfile tmpcert.$i.pem $X) \
			
		else
			echo "Parent Valid? This is the trust anchor"
		fi
		echo "#############################"
	fi
	((i++))
done
rm -f tmpcert.*.pem $TFILE
