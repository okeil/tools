#!/bin/bash
for X in output/*.xml; do 
	/usr/local/bin/scanpbnj -x $X
done
