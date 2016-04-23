#!/bin/bash
ret=fail
if [ $3 -eq 1 ]; then
	ret=$(echo Q | timeout 5 openssl s_client -servername "${1-`hostname`}" -connect "${1-`hostname`}:$2" -ssl3 2> /dev/null)
else
	ret=$(echo Q | timeout 5 openssl s_client -connect "${1-`hostname`}:$2" -ssl3 2> /dev/null)
fi

if echo "${ret}" | grep -q 'Protocol.*SSLv3'; then
  if echo "${ret}" | grep -q 'Cipher.*0000'; then
    echo "SSLv3 disabled"
  else
    echo "SSLv3 enabled"
 fi
else
  echo "SSL disabled or other error"
fi
