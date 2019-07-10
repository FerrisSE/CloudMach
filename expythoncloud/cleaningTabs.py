from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
# libraries needed for file reading and cleanup 
import os
import sys
import fileinput

# to clean up file from \t 


print ("Text to search for:")
textToSearch = '\t' 

print ("Text to replace it with:")
textToReplace = ''

print ("File to perform Search-Replace on:")
fileToSearch  = '1actualcsv.csv'


tempFile = open( fileToSearch, 'r+' )

for line in fileinput.input( fileToSearch ):
    if textToSearch in line :
        print('Match Found')
    else:
        print('Match Not Found!!')
    tempFile.write(line.replace( textToSearch, textToReplace ) )
tempFile.close()

input( '\n\n Press Enter to exit...' )

