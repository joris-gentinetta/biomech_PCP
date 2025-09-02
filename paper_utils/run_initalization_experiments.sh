#!/bin/bash

participants=("P_149" "P_238" "P_407" "P_426" "P_577" "P_668" "P_711" "P_950" "P7_453" "P6_820")

for participant in "${participants[@]}"
do
  if [ "$participant" = "P7_453" ]; then
    hand="Right"
  else
    hand="Left"
  fi
  echo "Running training for $participant with intact hand $hand"
  python s4_train.py --person_dir "$participant" --intact_hand "$hand" --config_name modular_initialization -hs
done