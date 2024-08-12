#!/usr/bin/env bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

####
## NOTE: For downloading Llama 3.1 please refer to https://github.com/meta-llama/llama-models/tree/main/models/llama3_1#download
####

set -ex  # Enable verbose output and exit on errors

# Dependency Check
if ! command -v wget &> /dev/null; then
    echo "wget is not installed. Please install it."
    exit 1
fi

read -p "Enter the URL from email: " PRESIGNED_URL
echo ""
read -p "Enter the list of models to download without spaces (8B,8B-instruct,70B,70B-instruct), or press Enter for all: " MODEL_SIZE
TARGET_FOLDER="."             # where all files should end up
mkdir -p "${TARGET_FOLDER}"

if [[ -z "$MODEL_SIZE" ]]; then
    MODEL_SIZE="8B,8B-instruct,70B,70B-instruct"
fi

echo "Downloading LICENSE and Acceptable Usage Policy"
wget --continue "${PRESIGNED_URL/'*'/'LICENSE'}" -O "${TARGET_FOLDER}/LICENSE" || { echo "Failed to download LICENSE. Exiting."; exit 1; }
wget --continue "${PRESIGNED_URL/'*'/'USE_POLICY'}" -O "${TARGET_FOLDER}/USE_POLICY" || { echo "Failed to download USE_POLICY. Exiting."; exit 1; }

for m in ${MODEL_SIZE//,/ }
do
    if [[ $m == "8B" ]] || [[ $m == "8b" ]]; then
        SHARD=0
        MODEL_FOLDER_PATH="Meta-Llama-3-8B"
        MODEL_PATH="8b_pre_trained"
    elif [[ $m == "8B-instruct" ]] || [[ $m == "8b-instruct" ]]; then
        SHARD=0
        MODEL_FOLDER_PATH="Meta-Llama-3-8B-Instruct"
        MODEL_PATH="8b_instruction_tuned"
    elif [[ $m == "70B" ]] || [[ $m == "70b" ]]; then
        SHARD=7
        MODEL_FOLDER_PATH="Meta-Llama-3-70B"
        MODEL_PATH="70b_pre_trained"
    elif [[ $m == "70B-instruct" ]] || [[ $m == "70b-instruct" ]]; then
        SHARD=7
        MODEL_FOLDER_PATH="Meta-Llama-3-70B-Instruct"
        MODEL_PATH="70b_instruction_tuned"
    else
        echo "Unknown model size: $m. Skipping."
        continue
    fi

    echo "Downloading ${MODEL_PATH}"
    mkdir -p "${TARGET_FOLDER}/${MODEL_FOLDER_PATH}"

    for s in $(seq -f "0%g" 0 ${SHARD})
    do
        wget --continue "${PRESIGNED_URL/'*'/"${MODEL_PATH}/consolidated.${s}.pth"}" -O "${TARGET_FOLDER}/${MODEL_FOLDER_PATH}/consolidated.${s}.pth" || { echo "Failed to download consolidated.${s}.pth. Exiting."; exit 1; }
    done

    wget --continue "${PRESIGNED_URL/'*'/"${MODEL_PATH}/params.json"}" -O "${TARGET_FOLDER}/${MODEL_FOLDER_PATH}/params.json" || { echo "Failed to download params.json. Exiting."; exit 1; }
    wget --continue "${PRESIGNED_URL/'*'/"${MODEL_PATH}/tokenizer.model"}" -O "${TARGET_FOLDER}/${MODEL_FOLDER_PATH}/tokenizer.model" || { echo "Failed to download tokenizer.model. Exiting."; exit 1; }
    wget --continue "${PRESIGNED_URL/'*'/"${MODEL_PATH}/checklist.chk"}" -O "${TARGET_FOLDER}/${MODEL_FOLDER_PATH}/checklist.chk" || { echo "Failed to download checklist.chk. Exiting."; exit 1; }

    echo "Checking checksums"
    CPU_ARCH=$(uname -m)
    if [[ "$CPU_ARCH" == "arm64" ]]; then
      (cd "${TARGET_FOLDER}/${MODEL_FOLDER_PATH}" && md5 checklist.chk) || { echo "Checksum verification failed. Exiting."; exit 1; }
    else
      (cd "${TARGET_FOLDER}/${MODEL_FOLDER_PATH}" && md5sum -c checklist.chk) || { echo "Checksum verification failed. Exiting."; exit 1; }
    fi

    echo "Download and verification of ${MODEL_PATH} completed successfully."
done
