#!./bin/python3.4
import socket, ssl, sys

#print(sys.version)

if len(sys.argv) != 3:
    print ("Usage: verify_hostname.py <hostname> <port>")
    sys.exit(2)

server_name = sys.argv[1]
server_port = sys.argv[2]

context = ssl.SSLContext(ssl.PROTOCOL_TLSv1) 
context.verify_mode = ssl.CERT_REQUIRED 
context.load_default_certs() 
context.check_hostname = True

sock = socket.create_connection((server_name, server_port)) 
# server_hostname is already used for SNI 
sslsock = context.wrap_socket(sock, server_hostname=server_name)

cert = 

try:
    ssl.match_hostname(sslsock.getpeercert(), server_name)
    print ("OK")
except:
    print("Unexpected error:", sys.exc_info()[0])
    raise
    #print ("INVALID")
