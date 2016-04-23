#!/usr/bin/python
## Python script for breadth first search applied to missionaries and cannibals
## Search is complete and optimal and complete assuming path cost is a non decreasing fucntion of the depth (which is true for missionaries and cannibalis)
# by mark culhane, 2011

import sys

n = input('how many missionaries?(cannibals = missionaries): ')
## With boat size set at 2 people, the max n is 3

frontier = [] #nodes that pass valid test but fail goal test but are yet to be exploded. Nodes reference their parent node. 
explored = [] #nodes that pass valid test but fail goal test. Nodes reference their parent
solution = [] #if a solution node is found, the path to the node is backward poulated here
solutionChild = []

def main():
## GENERAL TEXT AND FUNCTION CALLS
	try:
		rootNode = [n,n,'R']
	        print('****************************************************')
	        print('Initial State:')
		printState(rootNode)
		print('****************************************************')
		frontier.append([rootNode, rootNode])
		explored.append([rootNode, rootNode])
		if BFS(rootNode):
			print ('****************************************************')
			print ('SOLUTION FOUND! the solution is:')
			i = 0
			buildSolutionPath(solutionChild[0], rootNode)
			solution.reverse()
			solution.pop(0)
			for x in solution:
				if i == 0:
					print ('Initial State: ')
					printState(x[0])
					print()
				else:
					print ('step ', i, ': ')
					printState(x[0])
					print()
				i = i + 1
		else:
			print ('Failed to find a solution')
	except KeyboardInterrupt:
	        print ("Shutdown requested...exiting")
	except Exception, e:
        	print ("An unexpected exception was encountered: %s" % str(e))
	        sys.exit(1)
        raw_input('Hit a enter to quit')
	sys.exit('success')


def BFS(rootNode):
#### Breadth first search, frontier nodes are popped in FIFO order

	if goalTest([rootNode, rootNode]) == 1: ##check if initial state is solution
		solution[0] = rootNode
		return True
	count  = 0
	while frontier:
		curNode = [frontier[0][0], frontier[0][1]] #FIFO
		if curNode[0][2] == 'R':
			if popFrontierGoLeft(curNode): #Operators are divided to go left and go right		
				explored.append(curNode)
				return True
		else:
			popFrontierGoRight(curNode)
		explored.append(curNode)

		##########################################
		## Prints frontier so we can see search progress
		printFront = []
		for x in frontier:
			printFront.append(x[0])
		print ('Frontier Level ', count,': ', printFront)
		##########################################
		frontier.remove(curNode)
		count = count + 1
	return False 

def popFrontierGoLeft(node):
	## TWO MISSIONARIES GOING LEFT
	child = [[node[0][0],node[0][1],'L'],node[0]]
	child[0][0] = (child[0][0] - 2)
	if tryOpLeft(child):
		return True
        
	## ONE MISSIONARIES GOING LEFT
	child = [[node[0][0],node[0][1],'L'],node[0]]
	child[0][0] = (child[0][0] - 1)
	if tryOpLeft(child):
		return True

        ## ONE MISSIONARIES AND ONE CANN GOING LEFT
	child = [[node[0][0],node[0][1],'L'],node[0]]
	child[0][0] = (child[0][0] - 1)
	child[0][1] = (child[0][1] - 1)
	if tryOpLeft(child):
		return True

        ## ONE CANNI GOING LEFT
	child = [[node[0][0],node[0][1],'L'],node[0]]
	child[0][1] = (child[0][1] - 1)
	if tryOpLeft(child): 
		return True

	## TWO CANNIS GOING LEFT
	child = [[node[0][0],node[0][1],'L'],node[0]]
	child[0][1] = (child[0][1] - 2)
	if tryOpLeft(child):
		return True

	return False

def tryOpLeft(child):
	if checkValid(child):
		if goalTest(child):
			solutionChild.append(child)
			return True
		e = 0
		for n in explored:
			if n[0] == child[0]:
				e = 1
		if e == 0:	
			frontier.append(child)
	return False

def tryOpRight(child):
	if checkValid(child):
		e = 0
		for n in explored:
			if n[0] == child[0]:
				e = 1
		if e == 0:	
			frontier.append(child)
	return


def popFrontierGoRight(node): #dont need goal test
	## TWO MISSIONARIES GOING RIGHT
	child = [[node[0][0],node[0][1],'R'],node[0]]	
	child[0][0] = child[0][0] + 2
	tryOpRight(child)

        ## ONE MISSIONARIES GOING RIGHT
	child = [[node[0][0],node[0][1],'R'],node[0]]	
	child[0][0] = child[0][0] + 1
	tryOpRight(child)

        ## ONE MISSIONARIES AND ONE CANN GOING RIGHT
	child = [[node[0][0],node[0][1],'R'],node[0]]	
	child[0][0] = child[0][0] + 1
	child[0][1] = child[0][1] + 1
	tryOpRight(child)

        ## ONE CANNI GOING RIGHT
	child = [[node[0][0],node[0][1],'R'],node[0]]	
	child[0][1] = child[0][1] + 1
	tryOpRight(child)
 
	## TWO CANNIS GOING RIGHT
	child = [[node[0][0],node[0][1],'R'],node[0]]	
	child[0][1] = child[0][1] + 2
	tryOpRight(child)
	
	return

def goalTest(node):
	if node[0][0] == 0 and node[0][1] == 0:
		return True
	return False

def checkValid(node):
	if node[0][0] > n or node[0][1] > n:
		return False
	if node[0][0] < node[0][1] and node[0][0] > 0:
		return False
	if node[0][0] > node[0][1] and node[0][0] != n: #checkl for missionaries on right bank :)
		return False
	if node[0][0] < 0 or node[0][1] < 0:
		return False
	return True

def buildSolutionPath(node, rootNode):
	#builds solution from bottom up
	solution.append(node)
	while node[0] != rootNode:
		for n in explored:
			if n[0] == node[1]:
				solution.append(n)
				node = n[:]
def printState(node):
	if node[2] == 'R':
		print ('Miss[', n - node[0], '], cann[', n - node[1],'] |_______<B>|  Miss[', node[0], '], cann[', node[1],'] ')
	else:
		print ('Miss[', n - node[0], '], cann[', n - node[1],'] |<B>_______|  Miss[', node[0], '], cann[', node[1],'] ')
	return

if not raw_input('Hit a key to continue'):
	main()
