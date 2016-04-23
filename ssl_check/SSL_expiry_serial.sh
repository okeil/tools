#!/bin/bash
#########################################################################################
#											#
# DOES WHAT? 										#
#	- checks port 443 (and 700 for *epp* domains) for certs				#
#	- add certs expiring within $timeLeftThreshold to $expiryAlerts			#
#	- adds found certs to $outputFile						#
#	- adds all domain:port combos checked to $simpleResult				#
#	- removes output files more than 30 days old					#
#											#
# RUN: ./SSL_expiry_serial.sh								#
#											#
# DEPENDENCIES: 									#
#		- Files(webdnsDomains, portList, s_client_test.sh)			#
#		- all to be in same dir							#
#		- set normal path varibles (openssl,dig,find,rm etc)			#
# Ensure hashlinks for trustanchors							#
#	 for X in ./*.pem;do ln -s $X ./`openssl x509 -hash -noout -in $X`.0;done	#
#########################################################################################

cd /usr/local/ausregistry/nagios/ssl_check/

############################# FILE DEPS ##############################
manualDomains="./webdnsDomains"
portList=$(cat ./portList)
whiteListed="./whiteList"
s_client_test="./s_client_test.sh"
whiteListedNo=$(wc -l ${whiteListed} | cut -d " " -f1)
trustanchors="./trustanchors/"
######################################################################

########################## OUTPUT LOCATIONS ##########################
scanDate=`date +%d-%m-%Y`
outputDir=ssl_scan_output
outputFile=$outputDir/ssl_scan_results.$scanDate.csv
simpleResult=$outputDir/domainsChecked.$scanDate.txt
ticketThreshold=5184000 #60 days 60*60*24*90
nagiosAlertThreshold=604800 # 7 days 60*60*24*7
ticketItems=$outputDir/alerts/TICKET_ALERTS_$scanDate.csv
nagiosAlerts=$outputDir/alerts/NAGIOS_ALERTS_$scanDate.csv
errorLog=$outputDir/alerts/error_$scanDate.log
cat /dev/null > $simpleResult 
######################################################################

############################## NAGIOS VARS ##############################
NAGIOS_CRITICAL=2
NAGIOS_OK=0
NAGIOS_SERVICE="SSL Expiry Check - Public Services"
NAGIOS_HOSTNAME="SSL"
NAGIOS_SERVERS="mgmt1.mel.arsrs.local mgmt1.syd.arsrs.local"
NSCA_BINARY="/usr/sbin/send_nsca"
#########################################################################

############################## mgmt1 HTML vars #################################
MGMT_WWW_DIR="/var/www/html/ssl_check"
FULL_SCAN_PAGE="$MGMT_WWW_DIR/sslcheck_fulloutput.html"
DOMAINS_CHECKED="$MGMT_WWW_DIR/domains_checked.html"
EXPIRING_CERT_PAGE="$MGMT_WWW_DIR/expiringcert.html"
INVALID_CN_PAGE="$MGMT_WWW_DIR/invalidcn.html"
INVALID_PARENT_PAGE="$MGMT_WWW_DIR/invalidparent.html"
BAD_SIG_ALG_PAGE="$MGMT_WWW_DIR/badsigalg.html"
BAD_CIPHERS_PAGE="$MGMT_WWW_DIR/badciphers.html"
ALL_ISSUES_PAGE="$MGMT_WWW_DIR/allissues.html"
RESULT_LINKS='# OTHER RESULTS: <a href="./sslcheck_fulloutput.html">all_certs_found</a>, <a href="./domains_checked.html"> domains_checked</a>, <a href="./allissues.html"> all_issues</a>
		, <a href="./expiringcert.html"> expiring_certs</a>,<a href="./invalidcn.html"> invalid_cn</a>,<a href="./invalidparent.html"> invalid_chain</a>, 
		<a href="./badsigalg.html"> sha1_md5_sigalg</a>, <a href="./badciphers.html"> bad_ciphers</a>'
#########################################################################

touch $ticketItems $nagiosAlerts

if [ ! -d "$outputDir" ]; then
	mkdir "$outputDir" || exit 2
fi
if [ ! -d "$outputDir/alerts" ]; then
	mkdir "$outputDir/alerts" || exit 2
fi
if [ ! -d "$MGMT_WWW_DIR" ]; then
        mkdir "$MGMT_WWW_DIR" || exit 2
fi

#clean up temp file and outputs more than 30 days old
clean_up() {
	/bin/rm -f ./tmp_cert
	find $outputDir -type f -mtime +60 -print0 | xargs -0 /bin/rm -f
}


create_nagios_alert() {
	if [ $1 -ne 0 ] ; then
        	MSG="CRITICAL: SSL certificate at $2 expires in 7 Days OR has already Expired!"
	        NAGIOS_STATUS=${NAGIOS_CRITICAL}
	else
        	MSG="OK: No SSL Certificates Expiring in 7 Days. - [ ${whiteListedNo} Certificates Exempted ]"
	        NAGIOS_STATUS=${NAGIOS_OK}
	fi

	for nagios_host in ${NAGIOS_SERVERS} ; do
	        echo -e "${NAGIOS_HOSTNAME}\t${NAGIOS_SERVICE}\t${NAGIOS_STATUS}\t${MSG}" | ${NSCA_BINARY} -H $nagios_host > /dev/null
	done
}

# attempt to retrieve server cert
# $1 -> domain
# $2 -> port
test_ssl_connection() {
if ! grep --quiet "Testing: $1:$2," $simpleResult ; then
		timeout 5 "$s_client_test" $1 $2
	# Iterate through all certs pulled from server
		i=2
		for X in tmpcert.*.pem; do
			if openssl x509 -noout -in $X 2>/dev/null ; then
				alt_name=$(openssl x509 -text -noout -in $X |  sed -nr '/^ {12}X509v3 Subject Alternative Name/{n;s/^ *//p}')
				cn=$(openssl x509 -noout -subject -in $X | sed -e 's#.*CN=\(\)#\1#')
				cert_expiry_date=$(openssl x509 -noout -enddate -in $X \
						| awk -F= ' /notAfter/ { printf("%s\n",$NF); } ')
				SigAlg=$(openssl x509 -noout -text -in $X | grep -m1 "Signature Algorithm:" | head | awk '{print $3}')
				issuer=$(openssl x509 -noout -issuer -in $X | sed -n 's/^.*=\(.*\),*/\1/p')
				serial=$(openssl x509 -noout -serial -in $X | sed -n 's/^.*=\(.*\),*/\1/p')
                                
				########## CN + PARENT VALID? #########
                                # Check if domain falls into
				# this should use perl-libwww-perl-6
				wildcard=0
				stripped_cn=""
				if [[ $cn = \*\.* ]] ; then
        				stripped_cn=$(echo $cn | sed -n 's/[^.]*\.\(.*\),*/\1/p')
        				wildcard=1
				else 
				        stripped_cn=$cn
				fi
				stripped_domain=$(echo $1 | sed -n 's/[^.]*\.\(.*\),*/\1/p')
	
                                if [[ $1 == $stripped_cn ]] || [ $i -gt 2 ] || ([ $wildcard -eq 1 ] && [[ $stripped_domain == $stripped_cn ]]) ; then
					if [ $i -gt 2 ] ; then
						cn_valid="NA"
					else
						cn_valid="YES"
					fi
                               	        if [ -a tmpcert.$i.pem ]; then
						parent=$(openssl x509 -noout -subject -in tmpcert.$i.pem | sed -e 's#.*CN=\(\)#\1#')
                                        	parent_serial=$(openssl x509 -noout -serial -in tmpcert.$i.pem | sed -n 's/^.*=\(.*\),*/\1/p')
						if openssl verify -CApath $trustanchors -CAfile tmpcert.$i.pem $X >/dev/null ; then
        	                                       	parent_valid="YES"
	                	                else
        	                	        	parent_valid="NO"
                	                	fi
					else
						if openssl verify -CAfile $trustanchors* $X >/dev/null ; then
							parent_valid="YES to anchor"
						else
							parent_valid="NO to anchor"
						fi
					fi
	               		else
	                                if [[ $alt_name == *:$1* ]] ; then
						cn_valid="YES - Alt Name"
					else
						cn_valid="NO"
        	        	                parent_valid="Invalid CN"
					fi
				fi
				
				certVal="<td>$1</td><td>$2<td>$((i-1))</td><td>$cn</td><td>$serial</td><td>$SigAlg</td><td>$cert_expiry_date</td><td>$issuer</td><td>$parent</td><td>$parent_serial</td><td>$parent_valid</td><td>$cn_valid</td></tr>"
				echo "<tr>$certVal" >> $outputFile
				
				########## CREATE NAGIOS AND RT TICKETS ####################
				########## SHA1  leaf/intermediate? #########
				if [[ $SigAlg != "sha256WithRSAEncryption" ]] && [[ $SigAlg != "sha512WithRSAEncryption" ]] ; then
					if [[ $serial != $parent_serial ]] ; then
	                                	echo "<tr><td>Bad SigAlg</td>$certVal" >> $ticketItems
					fi
                                fi
                                if [[ $parent_valid == "NO" ]] ; then
                                	echo "<tr><td>Invalid Parent Offered</td>$certVal" >> $ticketItems
                                fi

				########## INVALID CN? #########
				if [[ "$cn_valid" == "NO" ]] ; then
					echo "<tr><td>Invalid CN</td>$certVal" >> $ticketItems
				fi
				########## EXPIRING? #########
				# Only add if the CN is valid
				if [[ "$cn_valid" == "YES" ]] && (grep --quiet "<td>$1</td>*<td>YES</td></tr>;" $outputFile || grep --quiet "<td>$1</td>*<td>YES - Alt Name</td></tr>;" $outputFile); then
					if ! openssl x509 -noout -checkend $ticketThreshold -in $X ; then
						echo "<tr><td>Expiry Threshold</td>$certVal" >> $ticketItems
					fi
					if ! openssl x509 -noout -checkend $nagiosAlertThreshold -in $X ; then
					#generate Nagios alert if serial is not in whitelist
						if ! grep --quiet $serial $whiteListed ; then
							#only add to nagiosAlerts if the serial is not already there
							if ! grep --quiet "$1:$2,$serial" $nagiosAlerts ; then
								create_nagios_alert 1 "$1:$2"
								((nagios_alert_count++))
								echo "$1:$2,$serial" >> $nagiosAlerts
							fi
						fi
					fi
				fi
				########## DONE WITH THIS CERT #########
				cert_found=1

			else
				cert_found=0
			fi
			((i++))
		done
		echo "Testing: $1:$2,cert_found=$cert_found<br>" | tee -a $simpleResult
		rm -f tmpcert.*.pem
fi
}
# $1 -> sub_domain
# calls test_ssl_connection
check_ssl() {
	if [ $# -gt 1 ]; then
		test_ssl_connection $1 $2
	else
		if [ $(echo $1 | tr -dc '*' | wc -c) -lt 1 ]; then
        		#only scanning 443 for non *epp*
	                testport=443
        	        test_ssl_connection $1 $testport
	                if [[ $1 == *"epp"* ]]; then
                		testport=700
                	        test_ssl_connection $1 $testport
        	        fi
		fi
	fi
}

# prep csv headers
#echo "Domain,port,serial,issuer,startdate,enddate,subject" > $outputFile
nagios_alert_count=0
echo "issue,domain,port,certlayer,CN,serial,sigAlg,expiry,issuer,offered_parent,parent_serial,parent_valid?,cn_valid?;" > $ticketItems

# RUN CHECKS ON ALL 'ARI' DOMAINS ON API.RESELLER.DISCOVERYDNS.COM
for DOMAIN in $(./reseller_zonelist.sh); do
	if ! grep --quiet "$DOMAIN" $whiteListed ; then
		check_ssl $DOMAIN
	fi
done

# RUN CHECKS ON ALL webdnsDomains items
while read DOMAIN; do
	if ! grep --quiet "$DOMAIN" $whiteListed ; then
		check_ssl $DOMAIN
	fi
done < $manualDomains

# Set nagios OK state if no alerts created
if [ $nagios_alert_count -eq 0 ] ; then
	echo "SENDING NAGIOS OK STATE" | tee -a $simpleResult
	create_nagios_alert 0
fi

# CREATE REVIEW PAGE FUNCTION
# TAKES ( 1 TITLE, 2 HTML FILE LOCATION, 3 FILTER) AS ARGS
make_results_page() {
	cols=0
	echo "<head><title>SSL_CHECK - $1</title></head>" > $2
	if [ $2 == $FULL_SCAN_PAGE ]; then
		cols=12
	else
		cols=13
	fi
	echo "<html><body><table><tr><th colspan="$cols">$1<th>" >> $2
	echo "<tr><td colspan="$cols"># SCAN DATE: $scanDate</td></tr>" >> $2
	echo "<tr><td colspan="$cols">$RESULT_LINKS</td></tr>" >> $2
	if [ $2 == $FULL_SCAN_PAGE ]; then
		echo "<tr><th>DOMAIN</th><th>PORT<th>LAYER</th><th>CN</th><th>SERIAL</th><th>SIGALG</th><th>EXPIRY</th><th>ISSUER</th><th>OFFERED PARENT</th><th>PARENT_SERIAL</th><th>PARENT VALID?</th><th>CN VALID?</th></tr>" >> $2
		/bin/cat $outputFile >> $2
	else
		echo "<tr><th>ISSUE</th><th>DOMAIN</th><th>PORT<th>LAYER</th><th>CN</th><th>SERIAL</th><th>SIGALG</th><th>EXPIRY</th><th>ISSUER</th><th>OFFERED PARENT</th><th>PARENT_SERIAL</th><th>PARENT VALID?</th><th>CN VALID?</th></tr>" >> $2
		grep "$3" $ticketItems >> $2
		if  grep --quiet "$3" $ticketItems ; then
			echo "EMAIL ALERT: $1"
		fi
	fi
	echo "</table>" >> $2
}

# CREATE DOMAINS CHECKED=
echo "# SCAN DATE: $scanDate <br />" > $DOMAINS_CHECKED
echo "<hr />$RESULT_LINKS<hr /><br />" >> $DOMAINS_CHECKED
/bin/cat $simpleResult >> $DOMAINS_CHECKED

# CREATE FULL SCAN RESULTS PAGE
make_results_page "ALL_CERTS_FOUND" "$FULL_SCAN_PAGE" ""
# CREATE ALL ISSUES PAGE
make_results_page "ALL ISSUES FOUND" "$ALL_ISSUES_PAGE" "tr"
# CREATE INVALID CN RESULTS PAGE
make_results_page "INVALID COMMON NAMES" "$INVALID_CN_PAGE" "<tr><td>Invalid CN<"

# CREATE BAD SIG ALG PAGE ON MGMT1
make_results_page "BAD SIGNATURE ALGS" "$BAD_SIG_ALG_PAGE" "<tr><td>Bad SigAlg<"

# CREATE BAD PARENT PAGE ON MGMT1
make_results_page "INVALID TRUST CHAIN" "$INVALID_PARENT_PAGE" "<tr><td>Invalid Parent Offered<"

# CREATE EXPIRING CERT PAGE ON MGMT1
make_results_page "EXPIRING CERTIFICATES" "$EXPIRING_CERT_PAGE" "<tr><td>Expiry Threshold<"

# CREATE BAD CIPHER PAGE
echo "# BAD CIPHERS DISCOVERED <br />" > $BAD_CIPHERS_PAGE
echo "<hr />$RESULT_LINKS<hr /><br />" >> $BAD_CIPHERS_PAGE
echo "<hr />-- TO BE IMPLEMENTED --<hr />" >> $BAD_CIPHERS_PAGE
# delete temp old results
clean_up
