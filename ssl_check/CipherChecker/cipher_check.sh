#!/bin/bash

# cipher_check.sh [domain] [port]
# output to stdout

if [ $# -ne 2 ]; then
	echo "USAGE: cipher_check.sh [domain] [port]"
	exit 1
fi
echo "#############################"
echo "#	$1 - CIPHER CHECK #"
nmap -p $2 --script ssl-enum-ciphers $1 #| grep broken
echo "#############################"
