#!/bin/bash
#########################################################################################
#																						#
# DOES WHAT? 																			#
#	- checks port 443 for certs															#
#	- add certs expiring within $timeLeftThreshold to $expiryAlerts						#
#	- adds found certs to $outputFile													#
#	- adds all domain:port combos checked to $simpleResult								#
#	- removes output files more than 30 days old										#
#																						#
# RUN: ./SSL_expiry_serial.sh															#
#																						#
#########################################################################################

############################# FILE DEPS ##############################
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${script_dir}
whiteListed="./whiteList"
s_client_test="./s_client_test.sh"
whiteListedNo=$(wc -l ${whiteListed} | cut -d " " -f1)
trustanchors="./trustanchors/"
manualEntries=$(cat ./manualEntries)
######################################################################

########################## OUTPUT LOCATIONS ##########################
scanDate=`date +%d-%m-%Y`
outputDir="ssl_scan_output"
outputFile=$outputDir/ssl_scan_results.$scanDate.csv
simpleResult=$outputDir/domainsChecked.$scanDate.txt
expiryAlertThreshold=2505000 # 29 days 60*60*24*29
ticketItems=$outputDir/alerts/TICKET_ALERTS_$scanDate.csv
######################################################################

############################## HTML vars #################################
WWW_DIR="/var/www/sites/ssl_site"
FULL_SCAN_PAGE="$WWW_DIR/index.html"
DOMAINS_CHECKED="$WWW_DIR/domains_checked.html"
EXPIRING_CERT_PAGE="$WWW_DIR/expiring_certs.html"
INVALID_CN_PAGE="$WWW_DIR/invalid_cns.html"
INVALID_PARENT_PAGE="$WWW_DIR/invalid_parents.html"
ALL_ISSUES_PAGE="$WWW_DIR/all_issues.html"
BAD_SIG_ALG_PAGE="$WWW_DIR/bad_sig_algs.html"
RESULT_LINKS='# OTHER RESULTS: <a href="./index.html">all_certs_found</a> - <a href="./domains_checked.html"> domains_checked</a> - <a href="./all_issues.html"> all_issues</a>
		 -  <a href="./expiring_certs.html"> expiring_certs</a> - <a href="./invalid_cns.html"> invalid_cn</a> - <a href="./invalid_parents.html"> invalid_chain</a> - <a href="./bad_sig_algs.html"> sha1_md5_sigalg</a>'
#########################################################################
cd $trustanchors
for X in ./*.pem;do ln -s $X ./`openssl x509 -hash -noout -in $X`.0;done
cd $script_dir

if [ ! -d "$outputDir/alerts" ]; then
	mkdir -p "$outputDir/alerts" || exit 2
fi
if [ ! -d "$WWW_DIR" ]; then
        mkdir "$WWW_DIR" || exit 2
fi

touch $ticketItems $simpleResult $outputFile

#clean up temp file and outputs more than 30 days old
clean_up() {
	/bin/rm -f ./tmp_cert
	find $outputDir -type f -mtime +60 -print0 | xargs -0 /bin/rm -f
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
				
				certVal="<div class=\"td\">$1</div><div class=\"td\">$2</div><div class=\"td\">$((i-1))</div><div class=\"td\">$(echo $cn | cut -c 1-25)</div><div class=\"td\">$(echo $SigAlg | cut -c 1-6)</div><div class=\"td\">${cert_expiry_date%????}</div><div class=\"td\">$(echo $issuer | cut -c 1-25)</div><div class=\"td\">$(echo $parent | cut -c 1-25)</div><div class=\"td\">$parent_valid</div><div class=\"td\">$cn_valid</div></div>"
				echo "<div class=\"row\">$certVal" >> $outputFile

                ########## ticketItems ####################
                ########## SHA1  leaf/intermediate? #########
                if [[ $SigAlg != "sha256WithRSAEncryption" ]] && [[ $SigAlg != "sha384WithRSAEncryption" ]] && [[ $SigAlg != "sha512WithRSAEncryption" ]] ; then
                        if [[ $serial != $parent_serial ]] ; then
                                                    echo "<div class=\"row\"><div class=\"td\">Bad SigAlg</div>$certVal" >> $ticketItems
                        fi
                fi
                if [[ $parent_valid == "NO" ]] ; then
                	echo "<div class=\"row\"><div class=\"td\">Invalid Parent Offered</div>$certVal" >> $ticketItems
                fi
                ########## INVALID CN? #########
                if [[ "$cn_valid" == "NO" ]] ; then
                	echo "<div class=\"row\"><div class=\"td\">Invalid CN</div>$certVal" >> $ticketItems
                fi
                ########## EXPIRING? #########
                # Only add if the CN is valid
                if [[ "$cn_valid" == "YES" ]]; then
                    if ! openssl x509 -noout -checkend $expiryAlertThreshold -in $X ; then
						if ! grep --quiet $serial $whiteListed ; then
	                    	echo "<div class=\"row\"><div class=\"td\">Expiry Threshold</div>$certVal" >> $ticketItems
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
	if [[ $1 == *":"* ]]; then
		IFS=: read hostname_t port_t <<<"${1}"
			test_ssl_connection $hostname_t $port_t
	else
		testport=443
        test_ssl_connection ${1/%?/} $testport
	fi
}

# RUN CHECKS ON ALL 'AO ROUTE 53' DOMAINS/SUBDOMAINS (THAT ARE A/AAAA/CNAME)
for DOMAIN in $(./list_external_domains.py); do
	if ! grep --quiet "$DOMAIN" $whiteListed ; then
		check_ssl $DOMAIN
	fi
done

for DOMAIN in $manualEntries; do
	check_ssl $DOMAIN
done

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
	echo "<html><head>
			<script src="https://ajax.googleapis.com/ajax/libs/jquery/1.12.0/jquery.min.js"></script>
        	<script src="theme.js"></script>
			<link rel="stylesheet" type="text/css" href="basic.css">
		</head><body><div class=\"results_box\"><div class=\"top_row\">$1</div>" >> $2
	echo "<div class=\"top_row\"># SCAN DATE: $scanDate</div>" >> $2
	echo "<div class=\"top_row\">$RESULT_LINKS</div></div><div class=\"results_box\">" >> $2
	if [ $2 == $FULL_SCAN_PAGE ]; then
		echo "<div class=\"row_head\"><div class=\"th\">DOMAIN</div><div class=\"th\">PORT</div><div class=\"th\">LAYER</div><div class=\"th\">CN</div><div class=\"th\">SIGALG</div><div class=\"th\">EXPIRY</div><div class=\"th\">ISSUER</div><div class=\"th\">OFFERED PARENT</div><div class=\"th\">PARENT VALID?</div><div class=\"th\">CN VALID?</div></div>" >> $2
		/bin/cat $outputFile >> $2
	else
		echo "<div class=\"row_head\"><div class=\"th\">ISSUE</div><div class=\"th\">DOMAIN</div><div class=\"th\">PORT</div><div class=\"th\">LAYER</div><div class=\"th\">CN</div><div class=\"th\">SIGALG</div><div class=\"th\">EXPIRY</div><div class=\"th\">ISSUER</div><div class=\"th\">OFFERED PARENT</div><div class=\"th\">PARENT VALID?</div><div class=\"th\">CN VALID?</div></div>" >> $2
		grep "$3" $ticketItems >> $2
	fi
	echo "</div></body></html>" >> $2
}

# CREATE DOMAINS CHECKED=
echo "# SCAN DATE: $scanDate <br />" > $DOMAINS_CHECKED
echo "<hr />$RESULT_LINKS<hr /><br />" >> $DOMAINS_CHECKED
/bin/cat $simpleResult >> $DOMAINS_CHECKED

# CREATE FULL SCAN RESULTS PAGE
make_results_page "ALL_CERTS_FOUND" "$FULL_SCAN_PAGE" ""
# CREATE ALL ISSUES PAGE
make_results_page "ALL ISSUES FOUND" "$ALL_ISSUES_PAGE" "<div class=\"row\">"
# CREATE INVALID CN RESULTS PAGE
make_results_page "INVALID COMMON NAMES" "$INVALID_CN_PAGE" "<div class=\"td\">Invalid CN<"

# CREATE BAD SIG ALG PAGE ON MGMT1
make_results_page "BAD SIGNATURE ALGS" "$BAD_SIG_ALG_PAGE" "<div class=\"td\">Bad SigAlg<"

# CREATE BAD PARENT PAGE ON MGMT1
make_results_page "INVALID TRUST CHAIN" "$INVALID_PARENT_PAGE" "<div class=\"td\">Invalid Parent Offered<"

# CREATE EXPIRING CERT PAGE ON MGMT1
make_results_page "EXPIRING CERTIFICATES" "$EXPIRING_CERT_PAGE" "<div class=\"td\">Expiry Threshold<"

# delete temp old results
clean_up
