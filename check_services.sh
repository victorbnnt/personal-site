#!/bin/bash

# Define color codes
BLACK='\033[0;30m'
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[0;37m'
NC='\033[0m' # No Color

# emoticon
CHECKMARK="✅"
CROSS="❌"

# TAXIFARE CHECK
output=$(ps aux | grep uvicorn | grep "8900")

# Check if the output is empty
if [ -n "$output" ]; then
  echo "${GREEN}${CHECKMARK} Taxifare api is running ${NC}"
else
  echo "> ${RED}${CROSS} Taxifare api is not running${NC}"
fi



# SAVINGS CHECK
output=$(ps aux | grep streamlit | grep "8901")

# Check if the output is empty
if [ -n "$output" ]; then
  echo "${GREEN}${CHECKMARK} Savings is running${NC}"
else
  echo "${RED}${CROSS} Savings is not running${NC}"
fi



# BLUEJAY CHECK
output=$(ps aux | grep "uwsgi" | grep "bluejay")

# Check if the output is empty
if [ -n "$output" ]; then
  echo "${GREEN}${CHECKMARK} Bluejay is running${NC}"
else
  echo "${RED}${CROSS} Bluejay is not running${NC}"
fi