from fastapi import HTTPException
from google.oauth2 import service_account
import os # Import os module to interact with environment variables
import logging # Import logging to provide information/warnings

# Assuming load_gcp_credentials is in config/Authentication/gcp.py
# and is imported correctly.
from config.Authentication.gcp import load_gcp_credentials


def get_gcp_credentials() -> service_account.Credentials:

    # Now call the original load_gcp_credentials function.
    # It will no longer see the problematic path if it was removed above.
    credentials = load_gcp_credentials()

    if not credentials:
        raise HTTPException(
            status_code=500,
            detail="Failed to load GCP credentials. Check your config/config.ini or environment variables.",
        )
    return credentials