#!/bin/bash

# Check if the script is run as root
if [ "$(id -u)" != "0" ]; then
   echo "This script must be run as root" 1>&2
   exit 1
fi

# Check if port argument is provided
if [ -z "$1" ]; then
   echo "Usage: $0 <port_number>"
   exit 1
fi

PORT=$1

# Find the PID(s) of the processes listening on the specified port
PID_LIST=$(lsof -t -i:$PORT | head -n 1)

# Check if any processes are listening on the specified port
if [ -z "$PID_LIST" ]; then
    echo "No processes are listening on port $PORT"
else
    echo "Processes listening on port $PORT have the following PID(s): $PID_LIST"
    # Kill the processes listening on the specified port
    kill "$PID_LIST"
    echo "Processes listening on port $PORT have been terminated"
fi