#!/bin/bash

# The command you want to schedule (use absolute paths)
CRON_CMD="/home/justinfields/Documents/github/spp_weis_price_forecast/scripts/model_retrain.sh"

# The cron schedule (e.g., run every 15 minutes)
CRON_JOB="5 0 * * 7 $CRON_CMD"

# Check if the command already exists in the crontab
if crontab -l 2>/dev/null | grep -qF "$CRON_CMD"; then
    echo "Cron job already exists. Skipping."
else
    echo "Adding cron job: $CRON_JOB"
    # Append the new job and install the combined crontab
    (crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -
fi
