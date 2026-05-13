#!/bin/bash

# Build trec_eval
cd /mnt/c/Users/kevin/Desktop/FAV/4.Rocnik/IR/sp/trec_eval-main/trec_eval-main
make

# Evaluate English data
echo "=== Evaluating English Data ==="
./trec_eval -q /mnt/c/Users/kevin/Desktop/FAV/4.Rocnik/IR/sp/data/eval_data_en/gold_relevancies.txt /mnt/c/Users/kevin/Desktop/FAV/4.Rocnik/IR/sp/data/evaluation/ranked_results_eval_data_en_2026-04-07_14-14-21.txt >> /mnt/c/Users/kevin/Desktop/FAV/4.Rocnik/IR/sp/data/evaluation/eval_results_en.txt

# Evaluate Czech data
echo "=== Evaluating Czech Data ==="
./trec_eval -q /mnt/c/Users/kevin/Desktop/FAV/4.Rocnik/IR/sp/data/eval_data_cs/gold_relevancies.txt /mnt/c/Users/kevin/Desktop/FAV/4.Rocnik/IR/sp/data/evaluation/ranked_results_eval_data_cs_2026-04-07_15-40-04.txt >> /mnt/c/Users/kevin/Desktop/FAV/4.Rocnik/IR/sp/data/evaluation/eval_results_cs.txt

echo "Done! Results saved to eval_results_en.txt and eval_results_cs.txt"
