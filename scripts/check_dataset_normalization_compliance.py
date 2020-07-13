import os
import sys

def exists(a, b):
	for i in a:
		if( b.find(i)!=-1 ):
			return True
	return False

dataset_filename = sys.argv[1]

lines = open( dataset_filename ).read().split("\n")

digits = []
others = []
i = 0

for line in lines:
	tokens = line.split("|")
	if(len(tokens)>1):
		if( any(char.isdigit() for char in tokens[1]) ):
			digits.append(i)
	i+=1

print( str(len(digits)) + "lines contain digits" )

invalid_character = [ '“', '’', '”' ]

for line in lines:
	tokens = line.split("|")
	if(len(tokens)>1):
		if( exists(invalid_character, tokens[1]) ):
			others.append(i)
	i+=1

print( str(len(others)) + "lines contain invalid characters" )