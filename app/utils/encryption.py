import os
from cryptography.fernet import Fernet, InvalidToken
from app.utils.logger import logger
from app.config.settings import settings



ENCRYPTION_KEY = settings.ENCRYPTION_KEY

# 2. Critical check: Ensure the key is set before the application runs
if not ENCRYPTION_KEY:
    raise ValueError("ENCRYPTION_KEY environment variable not set. Please generate a key and add it to your .env file.")

try:
    # 3. Initialize the Fernet cipher suite
    fernet = Fernet(ENCRYPTION_KEY.encode())
except (ValueError, TypeError) as e:
    raise ValueError(f"Invalid ENCRYPTION_KEY: {e}. Ensure it is a valid 32-byte URL-safe base64-encoded string.")

def encrypt_value(value: str) -> str:
    """Encrypts a string value using Fernet."""
    if not value:
        return ""
    try:
        # The value must be in bytes, so we encode it
        encrypted_bytes = fernet.encrypt(value.encode())
        # Return the encrypted bytes as a string for database storage
        return encrypted_bytes.decode()
    except Exception as e:
        logger.error(f"Encryption failed: {e}")
        # Depending on security policy, you might want to raise an exception
        # or return an empty string/None. Raising is often safer.
        raise ValueError("Failed to encrypt value.") from e

def decrypt_value(token: str) -> str:
    """Decrypts a Fernet token back to a string."""
    if not token:
        return ""
    try:
        # The token must be in bytes, so we encode it
        decrypted_bytes = fernet.decrypt(token.encode())
        # Return the decrypted bytes as a string
        return decrypted_bytes.decode()
    except InvalidToken:
        logger.error("Decryption failed: Invalid token or key.")
        # Return an empty string or None if the token is invalid/tampered with
        return ""
    except Exception as e:
        logger.error(f"An unexpected decryption error occurred: {e}")
        return ""