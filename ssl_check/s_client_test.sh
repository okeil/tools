TFILE="/tmp/$(basename $0).$$.tmp"
trap "rm -f $TFILE" 0
if [ $2 -eq 5222 ] || [ $2 -eq 5269 ]; then
	openssl s_client -showcerts -starttls xmpp -connect $1:$2 2>/dev/null </dev/null > $TFILE
elif [ $2 -eq 25 ] || [ $2 -eq 587 ] || [ $2 -eq 465 ]; then
	openssl s_client -showcerts -starttls smtp -connect $1:$2 2>/dev/null </dev/null > $TFILE
elif [ $2 -eq 110 ] || [ $2 -eq 995 ]; then
        openssl s_client -showcerts -starttls pop3 -connect $1:$2 2>/dev/null </dev/null > $TFILE
elif [ $2 -eq 143 ] || [ $2 -eq 993 ]; then
        openssl s_client -showcerts -starttls imap -connect $1:$2 2>/dev/null </dev/null > $TFILE
else
        openssl s_client -showcerts -servername $1 -connect $1:$2 2>/dev/null </dev/null > $TFILE
fi
awk '/-----BEGIN CERTIFICATE-----/ {p=1}p' $TFILE | tac | awk '/-----END CERTIFICATE-----/ {p=1}p' | tac | awk 'BEGIN {c=0;} /BEGIN CERT/{c++} { print > "tmpcert." c ".pem"}'
rm -f $TFILE
