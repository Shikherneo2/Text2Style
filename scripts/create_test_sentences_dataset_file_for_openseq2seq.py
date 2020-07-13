# Given a list of sentences and a reference sound file, creates a infer/test filelist that can be consumed by GST-Tacotron.

import sys

sentence_list_file = sys.argv[1]
reference_sound_file = sys.argv[2]
output_filename = sys.argv[3]

sentences_list = open(sentence_list_file, "r").read().split("\n")
sentences_list = [ i.strip() for i in sentences_list if i.strip()!="" ]

out_lines = []

for i,line in enumerate(sentences_list):
	out_line = []
	out_line.append( reference_sound_file )
	out_line.append( line )
	out_line.append( str(i+1) )
	out_lines.append( "|".join(out_line) )

outfile = open( output_filename, "w" )
outfile.write( "\n".join(out_lines) )
outfile.close()