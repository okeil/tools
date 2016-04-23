low Extended Euclidean algorithm for finding	#
# inverse multiplivative of e mod(t) - required for RSA #
# Mark Culhane 2011, straightIT.com
#########################################################

def findInverse(t, e): 
	x = 0
	y = 1
	lastX = 1
	lastY = 0
	a = t
	b = e
	while b != 0:
		quotient = (a / b)
		
		tmpB = b
		b = (a % b)
		a = tmpB
		
		tmpX = x
		x = (lastX - (quotient * x))
		lastX = tmpX
		
		tmpY = y
		y = (lastY - (quotient * y))
		lastY = tmpY
	return (t + lastY)

if __name__ == "__main__":
	print "Extended Euclidean Algorithm"
	e = long(raw_input("e (public key):"))
	totient = long(raw_input("phi(n):"))
	print "The inverse multiplicative of: ",e," mod(",totient,") is: ", findInverse(totient, e) 
