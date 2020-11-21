#!/bin/bash
FILE_NAME=$1
DELIMITER=$2
NUM_OF_COLUMNS=$(head -1 $FILE_NAME|tr $DELIMITER "\n"|wc -l)
#echo "File name: $FILE_NAME, delimiter=$DELIMITER"
#echo "num of columns: $NUM_OF_COLUMNS"
paste <(seq $NUM_OF_COLUMNS) <(head -1 $FILE_NAME|tr $DELIMITER "\n")

