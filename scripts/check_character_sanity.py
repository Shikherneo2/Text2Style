import os

f = open("/home/sdevgupta/mine/Text2Style/train_list_embed_text_mapping.txt")
lines = f.read().split("\n")

digits = []
numbers = []
i = 0

for line in lines:
	tokens = line.split("|")
	if(len(tokens)>1):
		if( any(char.isdigit() for char in tokens[1]) ):
			digits.append(i)
	i+=1

print(len(digits))
print(lines[digits[0]])

# “ with "
# ’ with '
# ” with "

ff = [ '“', '’', '”' ]

def exists(a, b):
	for i in a:
		if( b.find(i)!=-1 ):
			return True
	return False
		

for line in lines:
	tokens = line.split("|")
	if(len(tokens)>1):
		if( exists(ff, tokens[1]) ):
			digits.append(i)
	i+=1

print(len(digits))
print(lines[digits[0]])
print(lines[digits[1]])

