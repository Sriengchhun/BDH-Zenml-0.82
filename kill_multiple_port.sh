#!/bin/bash

# Check if the script is run as root
if [ "$(id -u)" != "0" ]; then
   echo "This script must be run as root" 1>&2
   exit 1
fi

# Check if port arguments are provided
if [ $# -eq 0 ]; then
   echo "Usage: $0 <port_number1> <port_number2> ..."
   exit 1
fi

# Iterate through the provided port numbers
for PORT in "$@"; do
    if [[ $PORT == *-* ]]; then
        IFS='-' read -r start_port end_port <<< "$PORT"

        for (( port=$start_port; port<=$end_port; port++ )); do

        # Find the PID(s) of the processes listening on the specified port
        PID_LIST=$(lsof -t -i:$port | head -n 4)

        # Check if any processes are listening on the specified port
        if [ -z "$PID_LIST" ]; then
            echo "No processes are listening on port $port"
        else
            echo "Processes listening on port $port have the following PID(s): $PID_LIST"
            # Kill the processes listening on the specified port
            # kill "$PID_LIST"
            # Kill all the processes listening on the specified port
            for PID in $PID_LIST; do
                kill $PID
            done
            echo "Processes listening on port $PORT have been terminated"
        fi
        done
    else
        PID_LIST=$(lsof -t -i:$PORT | head -n 4)

        # Check if any processes are listening on the specified port
        if [ -z "$PID_LIST" ]; then
            echo "No processes are listening on port $PORT"
        else
            echo "Processes listening on port $PORT have the following PID(s): $PID_LIST"
            # Kill the processes listening on the specified port
            # kill "$PID_LIST"
            # Kill all the processes listening on the specified port
            for PID in $PID_LIST; do
                kill $PID
            done
            echo "Processes listening on port $PORT have been terminated"
        fi
    fi
done