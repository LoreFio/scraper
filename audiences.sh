#!/bin/bash

cd ./code/

rm -r ../output/all_sites/*
rm -r ../output/all_but_renault/*
rm -r ../output/only_renault/*
rm -r ../output/brands/*

mkdir ../output/all_sites
mkdir ../output/all_but_renault
mkdir ../output/only_renault
mkdir ../output/brands

python audience_study.py -g all -l english
python audience_study.py -g all -l french 

python audience_study.py -g renault -l english
python audience_study.py -g renault -l french 

python audience_study.py -g brands -l english
python audience_study.py -g brands -l french 

python audience_study.py -g not_renault -l english
python audience_study.py -g not_renault -l french 

cd ..
