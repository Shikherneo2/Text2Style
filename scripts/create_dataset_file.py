import os
import sys

embedding_infer_file = sys.argv[1]
embedding_location = sys.argv[2]
output_file = sys.argv[3]

raw_lines = open(embedding_infer_file).read().split("\n")
lines = [ i.strip().split("|") for i in raw_lines if i.strip()!="" ]

out_lines = []
for line in lines:
	embed_filename = os.path.join( embedding_location, "embed-"+line[-1]+".npy" )
	line[-1] = embed_filename
	out_lines.append( "|".join( line ) )

fout = open(output_file, "w")
fout.write( "\n".join(out_lines) )
fout.close()