#!/bin/bash

# Bin 檔目錄
BIN_DIR=/home/allenzhuang0117/Parallel-Computing/in_outBin

# 定義不同的資料大小 (要和你產生的 input/output 檔名對應)
sizes=(10000 100000 500000 1000000 1500000)

# 定義要測試的 process 數
procs=(1 2 4 8 16 32)

# 執行程式
EXE=./baseline

for n in "${sizes[@]}"; do
    infile="${BIN_DIR}/input_${n}.bin"

    # 檢查檔案是否存在
    if [[ ! -f "$infile" ]]; then
        echo "❌ 缺少輸入檔案: $infile"
        continue
    fi

    for p in "${procs[@]}"; do
        outfile="${BIN_DIR}/output_${n}_${p}.bin"
        echo "▶️ Running n=$n with $p processes ..."
        mpirun -np $p $EXE $n "$infile" "$outfile" --verify --metrics
    done
done