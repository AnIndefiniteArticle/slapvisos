#!/bin/bash
# The purpose of this script is to manage the configuration, execution, and
# output of the analysis codes. First version runs imaging analysis only.

# Arguments to parse:
# -c CONFIG
# -y

# Store array for positional arguments (currently none)
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    # Config file to use
    -c|--config)
      CONFIG="$2"
      shift
      shift
      ;;
    # Skip interactive confirmation
    -y)
      CONFIRM="skip"
      shift
      ;;
    # error on unknown options
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    # save positional arguments in case we ever want to use them
    *)
      POSITIONAL_ARGS+=("$1")
      shift
      ;;
  esac
done

# Welcome statement
echo "Seasonal and Latitudinal Atmospheric Photochemical Variations from Infrared Stellar Occultations of Saturn's Atmosphere"
echo

# What is config.py currently linked to?
CURRENTCONFIG=$(ls -lF config.py | awk '{print $NF}' | sed -e "s/.py//g" | sed -e "s|events/||g")

# If CONFIG not passed in through argument, prompt for it
if [ "$CONFIG" = '' ]; then
  # Select a new config file
  echo "List of config files in events directory:"
  ls events | grep .py | sed -e "s/.py//g"
  read -p "Select event to run [$CURRENTCONFIG]: " CONFIG
fi
  
# Manage CONFIG file
echo
if [ "$CURRENTCONFIG" = "$CONFIG" ] || [ "$CONFIG" = '' ]; then
  CONFIG=$CURRENTCONFIG
  echo "using existing config.py symlink for event $CONFIG"
  if [ "$CURRENTCONFIG" = '' ]; then
    echo "existing config.py symlink not found"
    exit 2
  fi
else
  if [ "$CURRENTCONFIG" != '' ]; then
    echo "removing symlink to $CURRENTCONFIG"
    rm config.py
  fi
  if [ -e "events/$CONFIG.py" ]; then
    echo "creating symlink to events/$CONFIG.py"
    ln -s events/$CONFIG.py config.py
  else 
    echo "Invalid config file"
    exit 2
  fi
fi

# Interactive confirmation (skippable by commandline argument)
if [ "$CONFIRM" != 'skip' ]; then
  echo
  echo "Confirmation:"
  echo
  echo "Configuration set to:"
  cat config.py
  echo
  read -p "Is this correct? [Y/n]: " CONFIRM
  if [ "$CONFIRM" = 'n' ] || [ "$CONFIRM" = 'no' ] || [ "$CONFIRM" = 'N' ] || [ "$CONFIRM" = 'No' ]; then
    echo "User aborted"
    exit 3
  fi
fi

# make the CONFIG directory, if we need to
echo
if [ -d $CONFIG ]; then
  echo "$CONFIG directory exists"
else
  echo "creating directory $CONFIG"
  mkdir $CONFIG
fi

# Check if data file exists, and report
echo
if [ -f $CONFIG/$CONFIG.npy ]; then
  echo "$CONFIG/$CONFIG.npy data file exists. Will use this instead of re-reading CUB files to save time. Delete this file (or rename it) to force code to re-read CUB files from scratch."
else
  echo "Code will read in CUB files to create $CONFIG/$CONFIG.npy"
fi

# Create timestamped output directory
echo
OUTDIR=$CONFIG$(date --iso-8601="minutes" -u | sed -e "s/+00:00//")z
echo "Creating output directory $OUTDIR"
mkdir $CONFIG/$OUTDIR

# Create venv from requirements, if it doesn't already exist
if [ -d venv ]; then
  echo "using existing venv directory"
else
  python3 -m venv venv
  venv/bin/pip install -r requirements.txt
fi

# Save venv library version information
echo
echo "Saving library information to $CONFIG/$OUTDIR/requirements.txt"
venv/bin/pip freeze > $CONFIG/$OUTDIR/requirements.txt

# run the code, save the output to a logfile
echo
LOGFILE=$CONFIG/$OUTDIR/log
echo "Running imaginganalysis.py with these settings, and sending output to $LOGFILE"
venv/bin/python imaginganalysis.py $CONFIG/$OUTDIR | tee -a $LOGFILE

# echo success
echo "Run complete! See output in $CONFIG/$OUTDIR"
