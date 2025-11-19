# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash
set -xeuo pipefail # Exit immediately if a command exits with a non-zero status

FFMPEG_VERSION=7.0.1

for i in "$@"; do
    case $i in
        --FFMPEG_VERSION=?*) FFMPEG_VERSION="${i#*=}";;
        *) ;;
    esac
    shift
done

export DEBIAN_FRONTEND=noninteractive
apt-get update

# Install video runtime libraries
apt-get install -y \
    libmp3lame0 \
    libvpx7

# Get a list of all currently installed packages
dpkg -l | awk '{print $2}' | sort > /tmp/packages_before_ffmpeg_build.txt

# Install build libraries for video dependency
apt-get install -y \
    autoconf \
    automake \
    build-essential \
    cmake \
    libtool \
    libmp3lame-dev \
    libvpx-dev \
    pkg-config \
    vainfo \
    yasm

# Get a list of all packages installed after the build dependencies
dpkg -l | awk '{print $2}' | sort > /tmp/packages_after_ffmpeg_build.txt

# Use `comm` to find packages that are in the 'after' list but not in the 'before' list.
PACKAGES_TO_REMOVE=$(comm -13 /tmp/packages_before_ffmpeg_build.txt /tmp/packages_after_ffmpeg_build.txt | tr '\n' ' ')

echo $PACKAGES_TO_REMOVE

# Clean up temporary package lists
rm /tmp/packages_before_ffmpeg_build.txt /tmp/packages_after_ffmpeg_build.txt

# INSTALL FFMPEG
wget -O /tmp/ffmpeg-snapshot.tar.bz2 https://www.ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.bz2
tar xjvf /tmp/ffmpeg-snapshot.tar.bz2 -C /tmp/
cd /tmp/ffmpeg-${FFMPEG_VERSION}
PATH="/usr/local/cuda/bin:$PATH" ./configure \
    --prefix=/usr/local \
    --extra-libs=-lpthread \
    --extra-libs=-lm \
    --disable-static \
    --enable-shared \
    --enable-libmp3lame \
    --enable-libvpx \
    --disable-doc \
    --disable-debug
make -j$(nproc)
make install
ldconfig

# Clean up build tools.
if [ -n "$PACKAGES_TO_REMOVE" ]; then
    echo "Purging build dependencies: $PACKAGES_TO_REMOVE"
    apt-get purge -y $PACKAGES_TO_REMOVE
    apt-get autoremove -y
fi

# Clean up FFmpeg source and temporary files
cd /
rm -rf /tmp/ffmpeg*
rm -rf /var/lib/apt/lists/
