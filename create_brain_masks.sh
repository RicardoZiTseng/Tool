#!/bin/bash
mkdir masks
mkdir bet_res
for var in {1..40}
do
	echo start process $var.nii.gz ...
	bet "$var".nii.gz"" "$var"_bet.nii.gz"" -m
	cp "$var"_bet.nii.gz"" bet_res
	cp "$var"_bet_mask.nii.gz"" masks
	echo end process of $var.nii.gz
done

rm -rf *_bet.nii.gz
rm -rf *_bet_mask.nii,gz

exit 0
