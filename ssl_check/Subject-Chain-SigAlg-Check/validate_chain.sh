#!/bin/bash
# USAGE: validate_chain.sh <domain> <port>

############################# FILE DEPS ##############################
s_client_test="./s_client_test.sh"
trustAnchors="./trustanchors/*"
######################################################################
invalid_alg=0
clean_up() {
        rm -f tmpcert.*.pem
}

# attempt to retrieve server cert
# $1 -> domain
# $2 -> port
test_ssl_connection() {
	timeout 5 "$s_client_test" $1 $2
	# Iterate through all certs pulled from server
	i=2
	for X in tmpcert.*.pem; do
		if openssl x509 -noout -in $X 2>/dev/null ; then
			alt_name=$(openssl x509 -text -noout -in $X |  sed -nr '/^ {12}X509v3 Subject Alternative Name/{n;s/^ *//p}')
			cn=$(openssl x509 -noout -subject -in $X | sed -e 's#.*CN=\(\)#\1#')
			echo "#################################################################################"
			echo "CERTIFICATE OFFERED LAYER: $((i-1))"
			echo "CN= $cn"
			cert_expiry_date=$(openssl x509 -noout -enddate -in $X \
					| awk -F= ' /notAfter/ { printf("%s\n",$NF); } ')
			SigAlg=$(openssl x509 -noout -text -in $X | grep -m1 "Signature Algorithm:" | head | awk '{print $3}')
			issuer=$(openssl x509 -noout -issuer -in $X | sed -n 's/^.*=\(.*\),*/\1/p')
			serial=$(openssl x509 -noout -serial -in $X | sed -n 's/^.*=\(.*\),*/\1/p')                     
			
			########## CN + PARENT VALID? #########
			# this should use perl-libwww-perl-6 if available
			wildcard=0
			stripped_cn=""
			if [[ $cn == \*\.* ]] ; then
       				stripped_cn=$(echo $cn | sed -n 's/[^.]*\.\(.*\),*/\1/p')
       				wildcard=1
			else 
			        stripped_cn=$cn
			fi
			stripped_domain=$(echo $1 | sed -n 's/[^.]*\.\(.*\),*/\1/p')
                        if [[ $1 == $stripped_cn ]] || [ $i -gt 2 ] || ([ $wildcard -eq 1 ] && [[ $stripped_domain == $stripped_cn ]]) ; then
                               	cn_valid="YES"
				if [ $i -gt 2 ] ; then
					cn_valid="NA"
				else
	                                echo "CN VALID: $1 fits $cn"
				fi
                      	        if [ -a tmpcert.$i.pem ]; then
					parent=$(openssl x509 -noout -subject -in tmpcert.$i.pem | sed -e 's#.*CN=\(\)#\1#')
                                       	parent_serial=$(openssl x509 -noout -serial -in tmpcert.$i.pem | sed -n 's/^.*=\(.*\),*/\1/p')
					if openssl verify -CAfile tmpcert.$i.pem $X >/dev/null ; then
                                               	parent_valid="YES"
	               	                else
                        	        	parent_valid="NO"
               	                	fi
				else
					if openssl verify -CAfile $trustAnchors $X >/dev/null ; then
						echo "VALIDATED TO TRUST ANCHORS"
						parent_valid="YES to anchor"
					else
						parent_valid="NO to anchor"
					fi
				fi
	       		else
				if [[ $alt_name == *:$1* ]] ; then
					cn_valid="alt name ok"
					echo "ALT NAME VALID: $1 to $alt_name"
				else
                                        echo "CN INVALID: $1 to $cn"
					echo "ALT INVALID: $1 to $alt_name"
					cn_valid="NO"
               	               		parent_valid="Invalid CN"
				fi
			fi
				
				
			########## CREATE NAGIOS AND RT TICKETS ####################
			########## SHA1  leaf/intermediate? #########
			if [[ $SigAlg != "sha256WithRSAEncryption" ]] && [[ $SigAlg != "sha512WithRSAEncryption" ]] ; then
	                        echo "## ISSUE FOUND ##"
				echo "******** $SigAlg signature alg offered by server *************"
				echo "$1:$2, cert: $cn, $SigAlg"
				invalid_alg=1
			fi
			if [[ $parent_valid == "NO" ]] || [[ $parent_valid == "NO to anchor" ]] ; then
                        	echo "******** Invalid Parent Offered ************* - $1:$2, cert: $cn"
				clean_up; exit 1
                        fi
			########## INVALID CN? #########
			if [[ "$cn_valid" == "NO" ]] ; then
				echo "******* Invalid CN found *********** - $1:$2, cert: $cn"
				clean_up; exit 1
			fi
			((i++))
			if [ $invalid_alg -eq 0 ] ; then
				echo "OK!"
			fi
			echo "#################################################################################"
		fi
	done
	clean_up
	if [[ $parent_valid == "YES to anchor" ]] && [ $invalid_alg -eq 0 ] ; then
		echo "*******************************************"
		echo "Certificate chain and subject appears valid"
		echo "*******************************************"
	fi
}
if [ $# -eq 1 ]; then
	echo no port provided, defaulting to 443
	test_ssl_connection $1 443
elif [ $# -eq 2 ]; then
	test_ssl_connection $1 $2
else
	echo "USAGE: validate_chain.sh <domain> <port>"
fi
