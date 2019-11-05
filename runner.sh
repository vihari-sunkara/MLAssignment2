#!/bin/bash
#ONE TIME STUFF
model_dir="model"

trn_ft_file="../assignmentData/trn_X_Xf.txt"
trn_lbl_file="../assignmentData/trn_X_Y.txt"
trn_ft_lbl_file="../assignmentData/trn_X_XY.txt"
tst_ft_file="../assignmentData/tst_X_Xf.txt"
tst_lbl_file="../assignmentData/tst_X_Y.txt"
score_file="../score_mat.txt"

#UNCOMMENT BELOW FOR FIRST TIME RUNNING TO FORMAT TRAIN DATA FOR C++
# cd assignmentData
# bash sedscript
# cd ..

#REPEAT STUFF

# mkdir -p $model_dir
# cd shallow
# ./bonsai_train $trn_ft_file $trn_lbl_file $trn_ft_lbl_file ../$model_dir \
#     -T 2 \
#     -s 0 \
#     -t 2 \
#     -w 110 \
#     -b 1.0 \
#     -c 0.5 \
#     -m 2 \
#     -f 0.1 \
#     -fcent 0 \
#     -k 0.0001 \
#     -siter 25 \
#     -q 0 \
#     -ptype 0 \
#     -ctype 0
# cd ..
# cp -f assignmentData/tst_X_Xf.txt data.X
# cp -f assignmentData/tst_X_Y.txt data.y


# python3 eval.py
#copy test data of the split again 
#cp -f assignmentData/tst_X_Xf.txt data.X
#cp -f original_data/data.y data.y
mkdir -p $model_dir
cd shallow
./bonsai_train $trn_ft_file $trn_lbl_file $trn_ft_lbl_file ../$model_dir \
    -T 2 \
    -s 0 \
    -t 2 \
    -w 115 \
    -b 1.0 \
    -c 1.5 \
    -m 2 \
    -f 0.15 \
    -fcent 0 \
    -k 0.0001 \
    -siter 20 \
    -q 0 \
    -ptype 0 \
    -ctype 0
cd ..
cp -f assignmentData/tst_X_Xf.txt data.X
cp -f assignmentData/tst_X_Y.txt data.y


python3 eval.py

mkdir -p $model_dir
cd shallow
./bonsai_train $trn_ft_file $trn_lbl_file $trn_ft_lbl_file ../$model_dir \
    -T 2 \
    -s 0 \
    -t 2 \
    -w 115 \
    -b 1.0 \
    -c 1.5 \
    -m 2 \
    -f 0.15 \
    -fcent 0 \
    -k 0.0001 \
    -siter 20 \
    -q 0 \
    -ptype 0 \
    -ctype 1
cd ..
cp -f assignmentData/tst_X_Xf.txt data.X
cp -f assignmentData/tst_X_Y.txt data.y


python3 eval.py



