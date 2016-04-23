#!/bin/bash
echo "Latest port changes as detected by scan1.ausregistry.com.au"
echo "########## Ports not in 53, 443, 80, 700, 43, 25###############"
/usr/local/bin/outputpbnj --query daily_sensitive --type tab
echo "###############################################################"
echo "---"
echo "############### New ports in last 48 hours ####################"
/usr/local/bin/outputpbnj --query daily_all --type tab
echo "###############################################################"
echo "---"
echo "###################### nmap scan log ##########################"
cat ./nmap_scan.log
echo "###############################################################"
#outputpbnj --query latestinfo_audit --type csv
