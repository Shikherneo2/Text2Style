import sys

sentences_list = open(sys.argv[1], "r").read().split("\n")
sentences_list = [ i.strip() for i in sentences_list if i.strip()!="" ]

reference_sound_file = sys.argv[2]

out_lines = []

for i,line in enumerate(sentences_list):
	out_line = []
	out_line.append( reference_sound_file )
	out_line.append( line )
	out_line.append( str(i+1) )
	out_lines.append( "|".join(out_line) )

outfile = open( sys.argv[3], "w" )
outfile.write( "\n".join(out_lines) )
outfile.close()