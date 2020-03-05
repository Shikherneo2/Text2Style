import os

f = open("/home/sdevgupta/mine/Text2Style/train_list_embed_text_mapping.txt")
fout = open("/home/sdevgupta/mine/Text2Style/train_list_embed_text_mapping_cleaned.txt", "w")
lines = f.read().split("\n")

digits = []
numbers = []
i = 0

fd = []
i = 0
out_tokens = []
for line in lines:
	tokens = line.split("|")
	if(len(tokens)<=1):
		continue
	text = tokens[1]
	if(text[-1]==";"):
		i+=1
		text = text[:-1]
		text = text+"."
	out_tokens.append( [ tokens[0], text, tokens[2] ] )

print(len(out_tokens))
out_lines = "\n".join(["|".join(token) for token in out_tokens])
fout.write(out_lines)
fout.close()
