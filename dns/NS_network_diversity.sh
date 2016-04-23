#!/bin/bash
# USAGE: NS_network_diversity.sh <zone to check> $<resolver, 8.8.8.8>

# IANA requirements: https://www.iana.org/help/nameserver-requirements

count_autonomous_systems() {
    TFILE="/tmp/$(basename $0).$$.tmp"
    trap "rm -f $TFILE" 0
    resolver=$2 zone=$1 type=$3
    echo #####################################
    echo --- ZONE: $zone ---
    echo "NAMESERVERS: "
    ns_name_list=$(dig @$resolver $zone NS +short)
    ns_ip_list=""
    echo "- $type -"
    for ns_rec in $ns_name_list; do
        echo "-- $ns_rec --"
        ns_ip=$(dig @$resolver $ns_rec $type +short)
        whois -h whois.cymru.com $ns_ip | tee -a $TFILE
        echo "-----------------"
    done
    echo "Number of autonomous systems $type: " $(cat $TFILE | awk '{print $1}' | grep -v 'AS' | sort | uniq -c | wc -l)
    echo #####################################
    rm -f $TFILE
}

if [ $# -eq 1 ]; then
    count_autonomous_systems $1 8.8.8.8 A
    count_autonomous_systems $1 8.8.8.8 AAAA
elif [ $# -eq 2 ]; then
    count_autonomous_systems $1 $2 A
    count_autonomous_systems $1 $2 AAAA
else
    echo "USAGE: NS_network_diversity.sh <zone to check> <resolver>"
fi	
