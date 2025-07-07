"""Service for handling Claude Code CLI credentials."""

import json
import logging
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class OAuthToken(BaseModel):
    """OAuth token information from Claude credentials."""

    access_token: str = Field(..., alias="accessToken")
    refresh_token: str = Field(..., alias="refreshToken")
    expires_at: int = Field(..., alias="expiresAt")
    scopes: list[str] = Field(default_factory=list)
    subscription_type: str | None = Field(None, alias="subscriptionType")

    @property
    def is_expired(self) -> bool:
        """Check if the token is expired."""
        now = datetime.now(UTC).timestamp() * 1000  # Convert to milliseconds
        return now >= self.expires_at

    @property
    def expires_at_datetime(self) -> datetime:
        """Get expiration as datetime object."""
        return datetime.fromtimestamp(self.expires_at / 1000, tz=UTC)


class OrganizationInfo(BaseModel):
    """Organization information from OAuth API."""

    uuid: str
    name: str


class AccountInfo(BaseModel):
    """Account information from OAuth API."""

    uuid: str
    email_address: str


class UserProfile(BaseModel):
    """User profile information from Anthropic OAuth API."""

    organization: OrganizationInfo | None = None
    account: AccountInfo | None = None


class ClaudeCredentials(BaseModel):
    """Claude credentials from the credentials file."""

    claude_ai_oauth: OAuthToken = Field(..., alias="claudeAiOauth")


class CredentialsService:
    """Service for managing Claude CLI credentials."""

    # Standard credential file locations
    CREDENTIAL_PATHS = [
        Path.home() / ".claude" / ".credentials.json",
        Path.home() / ".config" / "claude" / ".credentials.json",
    ]

    # OAuth constants
    OAUTH_BETA_VERSION = "oauth-2025-04-20"
    TOKEN_URL = "https://console.anthropic.com/v1/oauth/token"
    AUTHORIZE_URL = "https://claude.ai/oauth/authorize"
    CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
    REDIRECT_URI = "http://localhost:54545/callback"
    SCOPES = ["org:create_api_key", "user:profile", "user:inference"]

    @classmethod
    def find_credentials_file(
        cls, custom_paths: list[Path] | None = None
    ) -> Path | None:
        """Find the Claude credentials file in standard or custom locations.

        Args:
            custom_paths: Optional list of custom paths to search instead of standard paths

        Returns:
            Path to the credentials file if found, None otherwise
        """
        search_paths = custom_paths if custom_paths else cls.CREDENTIAL_PATHS

        logger.info("Searching for Claude credentials file...")
        for path in search_paths:
            logger.debug(f"Checking: {path}")
            if path.exists() and path.is_file():
                logger.info(f"Found credentials file at: {path}")
                return path
            else:
                logger.debug(f"Not found: {path}")

        logger.warning("No credentials file found in any searched locations:")
        for path in search_paths:
            logger.warning(f"  - {path}")
        return None

    @classmethod
    def load_credentials(
        cls, custom_paths: list[Path] | None = None
    ) -> ClaudeCredentials | None:
        """Load and parse Claude credentials from file.

        Args:
            custom_paths: Optional list of custom paths to search instead of standard paths

        Returns:
            Parsed credentials if found and valid, None otherwise
        """
        cred_file = cls.find_credentials_file(custom_paths)
        if not cred_file:
            logger.info(
                "No credentials file found - OAuth authentication not available"
            )
            return None

        try:
            logger.info(f"Loading credentials from: {cred_file}")
            with cred_file.open() as f:
                data = json.load(f)

            credentials = ClaudeCredentials.model_validate(data)

            # Log credential details (safely)
            oauth_token = credentials.claude_ai_oauth
            logger.info("Successfully loaded credentials:")
            logger.info(f"  - Subscription type: {oauth_token.subscription_type}")
            logger.info(f"  - Token expires at: {oauth_token.expires_at_datetime}")
            logger.info(f"  - Token expired: {oauth_token.is_expired}")
            logger.info(f"  - Scopes: {oauth_token.scopes}")

            return credentials

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse credentials file {cred_file}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error loading credentials from {cred_file}: {e}")
            return None

    @classmethod
    def validate_credentials(
        cls, custom_paths: list[Path] | None = None
    ) -> dict[str, str | bool | list[str] | None]:
        """Validate Claude credentials and return status information.

        Args:
            custom_paths: Optional list of custom paths to search instead of standard paths

        Returns:
            Dictionary with validation results including:
            - valid: Whether credentials are found and valid
            - expired: Whether the token is expired
            - subscription_type: The subscription type if available
            - expires_at: Token expiration datetime
            - error: Error message if validation failed
        """
        try:
            credentials = cls.load_credentials(custom_paths)
            if not credentials:
                return {
                    "valid": False,
                    "error": "No credentials file found in ~/.claude/credentials.json or ~/.config/claude/credentials.json",
                }

            oauth_token = credentials.claude_ai_oauth

            return {
                "valid": True,
                "expired": oauth_token.is_expired,
                "subscription_type": oauth_token.subscription_type,
                "expires_at": oauth_token.expires_at_datetime.isoformat(),
                "scopes": oauth_token.scopes,
            }

        except Exception as e:
            logger.exception("Error validating credentials")
            return {
                "valid": False,
                "error": str(e),
            }

    @classmethod
    def save_credentials(
        cls, credentials: ClaudeCredentials, custom_paths: list[Path] | None = None
    ) -> bool:
        """Save updated credentials back to file.

        Args:
            credentials: Updated credentials to save
            custom_paths: Optional list of custom paths to search instead of standard paths

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            cred_file = cls.find_credentials_file(custom_paths)

            # If no existing file found and we have custom paths, use the first one
            if not cred_file and custom_paths:
                cred_file = custom_paths[0]
                # Ensure parent directory exists
                cred_file.parent.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Creating new credential file at: {cred_file}")
            elif not cred_file:
                logger.error("No credentials file found to update")
                return False

            # Convert to dict with proper aliases
            data = credentials.model_dump(by_alias=True)

            with cred_file.open("w") as f:
                json.dump(data, f, indent=2)

            logger.debug("Successfully saved updated credentials")
            return True

        except Exception as e:
            logger.error(f"Error saving credentials: {e}")
            return False

    @classmethod
    async def refresh_token(
        cls, custom_paths: list[Path] | None = None
    ) -> tuple[str, ClaudeCredentials] | tuple[None, None]:
        """Refresh the OAuth token using the refresh token.

        Args:
            custom_paths: Optional list of custom paths to search instead of standard paths

        Returns:
            Tuple of (new_access_token, updated_credentials) if successful,
            (None, None) if refresh failed
        """
        try:
            credentials = cls.load_credentials(custom_paths)
            if not credentials:
                return None, None

            refresh_token = credentials.claude_ai_oauth.refresh_token

            headers = {
                "Content-Type": "application/json",
                "anthropic-beta": cls.OAUTH_BETA_VERSION,
                "User-Agent": "Claude-Code/1.0.43",
            }

            data = {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": cls.CLIENT_ID,
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    cls.TOKEN_URL,
                    headers=headers,
                    json=data,
                    timeout=30.0,
                )

                if response.status_code == 200:
                    result = response.json()

                    # Update the OAuth token with new values
                    # Calculate expires_at from expires_in (seconds)
                    expires_in = result.get("expires_in")
                    expires_at = None
                    if expires_in:
                        expires_at = int(
                            (datetime.now(UTC).timestamp() + expires_in) * 1000
                        )  # Convert to milliseconds

                    oauth_data = {
                        "accessToken": result.get("access_token"),
                        "refreshToken": result.get("refresh_token", refresh_token),
                        "expiresAt": expires_at,
                        "scopes": result.get("scope", "").split()
                        if result.get("scope")
                        else credentials.claude_ai_oauth.scopes,
                        "subscriptionType": credentials.claude_ai_oauth.subscription_type,
                    }

                    # Create updated credentials
                    updated_credentials = ClaudeCredentials(
                        claudeAiOauth=OAuthToken(**oauth_data)
                    )

                    # Save the updated credentials
                    if cls.save_credentials(updated_credentials, custom_paths):
                        logger.debug("Successfully refreshed and saved OAuth token")
                        return (
                            updated_credentials.claude_ai_oauth.access_token,
                            updated_credentials,
                        )
                    else:
                        logger.error("Failed to save refreshed token")
                        return None, None

                else:
                    logger.error(
                        f"Failed to refresh token: {response.status_code} - {response.text}"
                    )
                    return None, None

        except Exception as e:
            logger.exception("Error refreshing token")
            return None, None

    @classmethod
    async def fetch_user_profile(
        cls, access_token: str, custom_paths: list[Path] | None = None
    ) -> UserProfile | None:
        """Fetch user profile using OAuth token.

        This makes a call to refresh the token which also returns
        organization and account information.

        Args:
            access_token: Current access token
            custom_paths: Optional list of custom paths to search instead of standard paths

        Returns:
            UserProfile if successful, None otherwise
        """
        try:
            credentials = cls.load_credentials(custom_paths)
            if not credentials:
                return None

            refresh_token = credentials.claude_ai_oauth.refresh_token

            headers = {
                "Content-Type": "application/json",
                "anthropic-beta": cls.OAUTH_BETA_VERSION,
                "User-Agent": "Claude-Code/1.0.43",
            }

            data = {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": cls.CLIENT_ID,
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    cls.TOKEN_URL,
                    headers=headers,
                    json=data,
                    timeout=30.0,
                )

                if response.status_code == 200:
                    result = response.json()

                    # Update credentials with refreshed token if available
                    if result.get("access_token"):
                        # Calculate expires_at from expires_in (seconds)
                        expires_in = result.get("expires_in")
                        expires_at = None
                        if expires_in:
                            expires_at = int(
                                (datetime.now(UTC).timestamp() + expires_in) * 1000
                            )  # Convert to milliseconds

                        oauth_data = {
                            "accessToken": result.get("access_token"),
                            "refreshToken": result.get("refresh_token", refresh_token),
                            "expiresAt": expires_at,
                            "scopes": result.get("scope", "").split()
                            if result.get("scope")
                            else credentials.claude_ai_oauth.scopes,
                            "subscriptionType": credentials.claude_ai_oauth.subscription_type,
                        }

                        updated_credentials = ClaudeCredentials(
                            claudeAiOauth=OAuthToken(**oauth_data)
                        )

                        # Save the updated credentials
                        cls.save_credentials(updated_credentials, custom_paths)

                    # Extract organization and account info
                    profile = UserProfile(
                        organization=OrganizationInfo(**result.get("organization", {}))
                        if result.get("organization")
                        else None,
                        account=AccountInfo(**result.get("account", {}))
                        if result.get("account")
                        else None,
                    )
                    return profile
                else:
                    logger.error(
                        f"Failed to fetch user profile: {response.status_code} - {response.text}"
                    )
                    return None

        except Exception as e:
            logger.exception("Error fetching user profile")
            return None

    @classmethod
    def get_access_token(cls) -> str | None:
        """Get the current access token if available and not expired.

        Returns:
            Access token string or None if not available/expired
        """
        credentials = cls.load_credentials()
        if not credentials:
            return None

        if credentials.claude_ai_oauth.is_expired:
            logger.warning("Access token is expired")
            return None

        return credentials.claude_ai_oauth.access_token

    @classmethod
    async def get_access_token_with_refresh(
        cls, custom_paths: list[Path] | None = None
    ) -> str | None:
        """Get the current access token, refreshing if necessary.

        Args:
            custom_paths: Optional list of custom paths to search instead of standard paths

        Returns:
            Access token string or None if not available/refresh failed
        """
        logger.info("Getting access token with refresh capability...")
        credentials = cls.load_credentials(custom_paths)
        if not credentials:
            logger.error("No credentials available for OAuth authentication")
            return None

        # Check token expiration status
        oauth_token = credentials.claude_ai_oauth
        logger.info("Token status check:")
        logger.info(f"  - Current time: {datetime.now(UTC)}")
        logger.info(f"  - Token expires: {oauth_token.expires_at_datetime}")
        logger.info(f"  - Is expired: {oauth_token.is_expired}")

        # If token is not expired, return it
        if not oauth_token.is_expired:
            logger.info("Using valid access token (not expired)")
            return oauth_token.access_token

        # Token is expired, try to refresh it
        logger.warning("Access token is expired, attempting to refresh...")
        new_token, updated_credentials = await cls.refresh_token(custom_paths)

        if new_token and updated_credentials:
            logger.info("Successfully refreshed access token")
            logger.info(
                f"New token expires: {updated_credentials.claude_ai_oauth.expires_at_datetime}"
            )
            return new_token
        else:
            logger.error("Failed to refresh access token - authentication unavailable")
            return None

    @classmethod
    async def login(cls, custom_paths: list[Path] | None = None) -> bool:
        """Perform OAuth login flow and save credentials.

        Args:
            custom_paths: Optional list of custom paths to save credentials to

        Returns:
            True if login successful, False otherwise
        """
        import base64
        import hashlib
        import secrets
        import urllib.parse
        import webbrowser
        from http.server import BaseHTTPRequestHandler, HTTPServer
        from threading import Thread
        from urllib.parse import parse_qs, urlparse

        # Generate state parameter for security
        state = secrets.token_urlsafe(32)

        # Generate PKCE parameters
        code_verifier = secrets.token_urlsafe(32)
        code_challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode()).digest())
            .decode()
            .rstrip("=")
        )

        authorization_code = None
        error = None

        class OAuthCallbackHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                nonlocal authorization_code, error

                # Ignore favicon requests
                if self.path == "/favicon.ico":
                    self.send_response(404)
                    self.end_headers()
                    return

                parsed_url = urlparse(self.path)
                query_params = parse_qs(parsed_url.query)

                # Check state parameter
                received_state = query_params.get("state", [None])[0]

                if received_state != state:
                    error = "Invalid state parameter"
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b"Error: Invalid state parameter")
                    return

                # Check for authorization code
                if "code" in query_params:
                    authorization_code = query_params["code"][0]
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"Login successful! You can close this window.")
                elif "error" in query_params:
                    error = query_params.get("error_description", ["Unknown error"])[0]
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(f"Error: {error}".encode())
                else:
                    error = "No authorization code received"
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b"Error: No authorization code received")

            def log_message(self, format: str, *args: Any) -> None:
                # Suppress HTTP server logs
                pass

        # Start local HTTP server for OAuth callback on port 54545
        server = HTTPServer(("localhost", 54545), OAuthCallbackHandler)
        server_thread = Thread(target=server.serve_forever)
        server_thread.daemon = True
        server_thread.start()

        try:
            # Build authorization URL
            auth_params = {
                "response_type": "code",
                "client_id": cls.CLIENT_ID,
                "redirect_uri": cls.REDIRECT_URI,
                "scope": " ".join(cls.SCOPES),
                "state": state,
                "code_challenge": code_challenge,
                "code_challenge_method": "S256",
            }

            auth_url = f"{cls.AUTHORIZE_URL}?{urllib.parse.urlencode(auth_params)}"

            logger.info("Opening browser for OAuth authorization...")
            logger.info(f"If browser doesn't open, visit: {auth_url}")

            # Open browser
            webbrowser.open(auth_url)

            # Wait for callback (with timeout)
            import time

            timeout = 300  # 5 minutes
            start_time = time.time()

            while authorization_code is None and error is None:
                if time.time() - start_time > timeout:
                    error = "Login timeout"
                    break
                time.sleep(0.1)

            if error:
                logger.error(f"OAuth login failed: {error}")
                return False

            if not authorization_code:
                logger.error("No authorization code received")
                return False

            # Exchange authorization code for tokens
            token_data = {
                "grant_type": "authorization_code",
                "code": authorization_code,
                "redirect_uri": cls.REDIRECT_URI,
                "client_id": cls.CLIENT_ID,
                "code_verifier": code_verifier,
                "state": state,  # Include state parameter as shown in Go example
            }

            headers = {
                "Content-Type": "application/json",
                "anthropic-beta": cls.OAUTH_BETA_VERSION,
                "User-Agent": "Claude-Code/1.0.43",
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    cls.TOKEN_URL,
                    headers=headers,
                    json=token_data,  # Back to JSON format like refresh method
                    timeout=30.0,
                )

                if response.status_code == 200:
                    result = response.json()

                    # Calculate expires_at from expires_in
                    expires_in = result.get("expires_in")
                    expires_at = None
                    if expires_in:
                        expires_at = int(
                            (datetime.now(UTC).timestamp() + expires_in) * 1000
                        )

                    # Create credentials object
                    oauth_data = {
                        "accessToken": result.get("access_token"),
                        "refreshToken": result.get("refresh_token"),
                        "expiresAt": expires_at,
                        "scopes": result.get("scope", "").split()
                        if result.get("scope")
                        else cls.SCOPES,
                        "subscriptionType": result.get("subscription_type", "unknown"),
                    }

                    credentials = ClaudeCredentials(
                        claudeAiOauth=OAuthToken(**oauth_data)
                    )

                    # Save credentials
                    if custom_paths:
                        # For custom paths, ensure directory exists
                        for path in custom_paths:
                            path.parent.mkdir(parents=True, exist_ok=True)

                    if cls.save_credentials(credentials, custom_paths):
                        logger.info("Successfully saved OAuth credentials")
                        return True
                    else:
                        logger.error("Failed to save OAuth credentials")
                        return False

                else:
                    logger.error(
                        f"Token exchange failed: {response.status_code} - {response.text}"
                    )
                    return False

        except Exception as e:
            logger.exception("Error during OAuth login")
            return False

        finally:
            # Stop the HTTP server
            server.shutdown()
            server_thread.join(timeout=1)
