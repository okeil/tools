#Simple Python implementation of the Miller-Rabin Test	#
# Input 'n' to test if it is probably a prime		#
# Mark Culhane 2011, straightIT.com			#
#########################################################
import random

def MRTest(n, iterations):  
#This is the implementation of the Miller-Rabin Algorithm
#See http://en.wikipedia.org/wiki/Miller%E2%80%93Rabin_primality_test
# inputs are n, number to be tested and k, number of iterations
	count = 0
	while (count < (iterations)):
		b = random.randint(1, (n-1))
		#Find q and the odd number k such that n - 1 = (2^q)k
		q = 1
		while testEven((n-1)/(2**q)):
			q += 1
		k = ((n-1)/(2**q))
		#Test if (b^k) mod n = 1 is TRUE
		if ((b**k) % (n))!= 1:
			print "Failed b^k mod n"
		#if there exists i in [0, (q-1)] such that b**(k(2**i)) mod n = -1	
			i = 0
			isPass = False
			while (i < q):
				if ((b**(k*(2**i)))%(n)) == (n-1):
					print "Passed b^k(2^i) mod n = n - 1!"
					isPass = True
				i = i + 1
			if isPass == False:
				print "Failed b^k mod n AND no i where b**(k*(2**i)) = n-1"
				return False
		count = count + 1 
	#If n ever fails a one of the two criteria above, it returns false, else return true
	return True

def testEven(testNumber):
#Inefficient method for testing odd/even
	if testNumber % 2 == 0:
		return True
	else:
		return False

def probEstimate(iterations):
#The probability of n being a prime number after i inconclusive tests is (1 ? ((1/4)**i)))
	probability = 1 - 0.2500000**iterations
	probability = probability * 100
	return probability


def getN():
	n = long(raw_input("n (is it prime?):"))
	return n

def getIterations():
	iterations = int(raw_input("Enter a k value (how many iterations):"))
	return iterations
if __name__ == "__main__":
	print "Miller-Rabin Algorithm"
	n = getN()
	while testEven(n):
		print n," is an even number, they aren't generally prime!"
		print "Try Again!"
		n = getN()
	iterations = getIterations()
	if MRTest(n, iterations):
		print "The number: ",n," is a PRIME"
		print "Iterations: ", probEstimate(iterations)
	else:
		print "The number: ",n," is COMPOSITE"
