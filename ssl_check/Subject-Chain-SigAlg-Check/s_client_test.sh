TFILE="/tmp/$(basename $0).$$.tmp"
#echo | openssl s_client -servername $1 -connect $1:$2 2>/dev/null
openssl s_client -showcerts -servername $1 -connect $1:$2 2>/dev/null </dev/null > $TFILE
awk '/-----BEGIN CERTIFICATE-----/ {p=1}p' $TFILE | tac | awk '/-----END CERTIFICATE-----/ {p=1}p' | tac | awk 'BEGIN {c=0;} /BEGIN CERT/{c++} { print > "tmpcert." c ".pem"}'
#awk 'BEGIN {c=0;} /BEGIN CERT/{c++} { print > "tmpcert." c ".pem"}' < $TFILE
rm -f $TFILE
