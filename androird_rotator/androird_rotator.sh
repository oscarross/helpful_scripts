#!/bin/bash

COUNTER=0
ALL_ROTATIONS=100

# Functions
show_help() {
    cat <<EOF
Usage: $0 [options]
EXAMPLE:
    $0 -n 30
OPTIONS:
   -h           Help
   -n           Number of rotations
EOF
}

show_install_info_android_platform_tools() {
    cat <<EOF
❌ Please install "android-platform-tools"

You can install by brew cask
"brew cask install android-platform-tools"
EOF
}

# Get params
while getopts "hn:" opt; do
    case "$opt" in
    h)
        show_help
        exit 0
        ;;
    n) ALL_ROTATIONS="$OPTARG" ;;
    *) shift ;;
    esac
done

if [[ $(command -v mogrify) == "" ]]; then
    show_install_info_android_platform_tools
    exit 1
fi

adb shell settings put system accelerometer_rotation 0

while [ $COUNTER -le $ALL_ROTATIONS ] ; do
    for i in `seq 0 3`
    do
        sleep 1
        echo ">> ROTATIONS: $COUNTER, STYLE: $i"
        adb shell settings put system user_rotation $i

        COUNTER=$[COUNTER + 1]
    done
done
