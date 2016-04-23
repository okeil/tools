echo | openssl s_client -showcerts -starttls xmpp -servername $1 -connect $1:$2 2>/dev/null
