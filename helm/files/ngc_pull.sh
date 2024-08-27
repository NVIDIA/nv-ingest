#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# To ensure we actually have an NGC binary, switch to full path if default is used
if [ "$NGC_EXE" = "ngc" ]; then
  NGC_EXE=$(which ngc)
fi

# check if ngc cli is truly available at this point
if [ ! -x "$NGC_EXE" ]; then
  echo "ngc cli is not installed or available!"
  exit 1
fi

# download the model
directory="${STORE_MOUNT_PATH}/${NGC_MODEL_NAME}_v${NGC_MODEL_VERSION}"
echo "Directory is $directory"
ready_file="$directory/.ready"
lock_file="$directory/.lock"

mkdir -p "$directory"
exec 200>"$lock_file"
{
  if ln -s "$lock_file" "$lock_file.locked"; then
    trap 'rm -f $lock_file.locked' EXIT
    if [ ! -e "$ready_file" ]; then
      $NGC_EXE registry model download-version --dest "$STORE_MOUNT_PATH" "${NGC_CLI_ORG}/${NGC_CLI_TEAM}/${NGC_MODEL_NAME}:${NGC_MODEL_VERSION}"
      # decrypt the model - if needed (conditions met)
      if [ -n "${NGC_DECRYPT_KEY:+''}" ] && [ -f "$directory/${MODEL_NAME}.enc" ]; then
        echo "Decrypting $directory/${MODEL_NAME}.enc"
        # untar if necessary
        if [ -n "${TARFILE:+''}" ]; then
          echo "TARFILE enabled, unarchiving..."
          openssl enc -aes-256-cbc -d -pbkdf2 -in "$directory/${MODEL_NAME}.enc" -out "$directory/${MODEL_NAME}.tar" -k "${NGC_DECRYPT_KEY}"
          tar -xvf "$directory/${MODEL_NAME}.tar" -C "$STORE_MOUNT_PATH"
          rm "$directory/${MODEL_NAME}.tar"
        else
          openssl enc -aes-256-cbc -d -pbkdf2 -in "$directory/${MODEL_NAME}.enc" -out "$directory/${MODEL_NAME}" -k "${NGC_DECRYPT_KEY}"
        fi
        rm "$directory/${MODEL_NAME}.enc"
      else
        echo "No decryption key provided, or encrypted file found. Skipping decryption.";
        if [ -n "${TARFILE:+''}" ]; then
          echo "TARFILE enabled, unarchiving..."
          tar -xvf "$directory/${NGC_MODEL_VERSION}.tar.gz" -C "$STORE_MOUNT_PATH"
          rm "$directory/${NGC_MODEL_VERSION}.tar.gz"
        fi
      fi
      touch "$ready_file"
      echo "Done dowloading"
    else
      echo "Download was already complete"
    fi;
    rm -f "$lock_file.locked"
  else
    while [ ! -e "$ready_file" ]
    do
      echo "Did not get the download lock. Waiting for the pod holding the lock to download the files."
      sleep 1
    done;
    echo "Done waiting"
  fi
}
ls -la "$directory"
