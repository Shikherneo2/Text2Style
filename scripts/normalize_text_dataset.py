# Run this on top of the default tacotron normalization.
# Text should express words, expression(? or !), pause lengths with "," and "-" and separate paraphrasing with "


############### Normalizing rules ######################################
# 1. all text ends in [".", "!", "?", '"']
# 2. If the end character is not one of allowed end characters, replaces it with .
# 3. replace ( , ) , -- and : with ,
# 4. remove '
# 5. trim whitespaces to single space


import os
import sys

replace_with = ","
non_allowed_chars = "'"
pause_chars = [ "(" , ")" , "--", ":"]

def replace_chars( txt, pause_chars, non_allowed_chars, replace_with ):
	for char in pause_chars:
		txt = txt.replace( char, "," )
	
	for char in non_allowed_chars:
		txt = txt.replace( char, "" )

	return txt


input_text_list = sys.argv[1]
output_text_list = sys.argv[2]

lines = open( input_text_list ).read().split("\n")
fout = open(output_text_list, "w")

out_tokens = []

for line in lines:
	tokens = line.split("|")
	if(len(tokens)<=1):
		continue
	text = tokens[1]
	
	if( text[-1].isalnum() ):
		text = text+"."

	if( text[-1]!="." and text[-1]!="!" and text[-1]!="?" and text[-1]!='"'):
		text = text[:-1]
		text = text+"."
	
	text = replace_chars( text, pause_chars, non_allowed_chars, replace_with )
	text = " ".join( [i for i in text.split(" ") if i.strip()!=""] )
	out_tokens.append( [ tokens[0], text, tokens[2] ] )


out_lines = "\n".join(["|".join(token) for token in out_tokens])
fout.write(out_lines)
fout.close()
