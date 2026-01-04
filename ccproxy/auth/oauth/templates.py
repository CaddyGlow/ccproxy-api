"""Centralized HTML templates for OAuth responses."""

from enum import Enum
from typing import Any

from fastapi.responses import HTMLResponse


class OAuthProvider(Enum):
    """OAuth provider types."""

    CLAUDE = "Claude"
    OPENAI = "OpenAI"
    GENERIC = "OAuth Provider"


class OAuthTemplates:
    """Centralized HTML templates for OAuth responses.

    This class provides consistent HTML responses across all OAuth providers,
    reducing code duplication and ensuring a uniform user experience.
    """

    # Base HTML template with common styling
    _BASE_TEMPLATE = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{title}</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
                padding: 20px;
                box-sizing: border-box;
            }}
            .container {{
                background: white;
                border-radius: 12px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                padding: 40px;
                max-width: 500px;
                width: 100%;
                text-align: center;
            }}
            h1 {{
                color: {header_color};
                margin: 0 0 20px 0;
                font-size: 28px;
                font-weight: 600;
            }}
            .icon {{
                font-size: 48px;
                margin-bottom: 20px;
            }}
            p {{
                color: #4a5568;
                font-size: 16px;
                line-height: 1.6;
                margin: 10px 0;
            }}
            .error-detail {{
                background-color: #fef2f2;
                border: 1px solid #fecaca;
                border-radius: 6px;
                padding: 12px;
                margin-top: 20px;
                color: #991b1b;
                font-family: 'Courier New', Courier, monospace;
                font-size: 14px;
                text-align: left;
                word-wrap: break-word;
            }}
            .success-message {{
                background-color: #f0fdf4;
                border: 1px solid #86efac;
                border-radius: 6px;
                padding: 12px;
                margin-top: 20px;
                color: #166534;
            }}
            .countdown {{
                color: #6b7280;
                font-size: 14px;
                margin-top: 20px;
            }}
            .action-hint {{
                color: #9ca3af;
                font-size: 14px;
                margin-top: 15px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            {content}
        </div>
        {script}
    </body>
    </html>
    """

    # Success content template
    _SUCCESS_CONTENT = """
        <div class="icon">✅</div>
        <h1>Authentication Successful!</h1>
        <p>You have successfully authenticated with {provider}.</p>
        <div class="success-message">
            Your credentials have been saved securely.
        </div>
        <p class="action-hint">You can close this window and return to the terminal.</p>
        <div class="countdown" id="countdown">This window will close automatically in 3 seconds...</div>
    """

    # Error content template
    _ERROR_CONTENT = """
        <div class="icon">❌</div>
        <h1>{title}</h1>
        <p>{message}</p>
        {error_detail}
        <p class="action-hint">You can close this window and try again.</p>
        <div class="countdown" id="countdown">This window will close automatically in 5 seconds...</div>
    """

    # Auto-close script
    _AUTO_CLOSE_SCRIPT = """
    <script>
        let seconds = {seconds};
        const countdownEl = document.getElementById('countdown');

        const updateCountdown = () => {{
            if (seconds > 0) {{
                countdownEl.textContent = `This window will close automatically in ${{seconds}} second${{seconds === 1 ? '' : 's'}}...`;
                seconds--;
                setTimeout(updateCountdown, 1000);
            }} else {{
                countdownEl.textContent = 'Closing window...';
                window.close();
            }}
        }};

        updateCountdown();

        // Try to close even if countdown doesn't work
        setTimeout(() => {{
            window.close();
        }}, {milliseconds});
    </script>
    """

    @classmethod
    def success(
        cls,
        provider: OAuthProvider = OAuthProvider.GENERIC,
        auto_close_seconds: int = 3,
        **kwargs: Any,
    ) -> HTMLResponse:
        """Generate success HTML response.

        Args:
            provider: OAuth provider name
            auto_close_seconds: Seconds before auto-closing window
            **kwargs: Additional template variables

        Returns:
            HTML response for successful authentication
        """
        content = cls._SUCCESS_CONTENT.format(provider=provider.value, **kwargs)

        script = cls._AUTO_CLOSE_SCRIPT.format(
            seconds=auto_close_seconds, milliseconds=auto_close_seconds * 1000
        )

        html = cls._BASE_TEMPLATE.format(
            title="Authentication Successful",
            header_color="#10b981",
            content=content,
            script=script,
        )

        return HTMLResponse(content=html, status_code=200)

    @classmethod
    def error(
        cls,
        error_message: str,
        title: str = "Authentication Failed",
        error_detail: str | None = None,
        status_code: int = 400,
        auto_close_seconds: int = 5,
        **kwargs: Any,
    ) -> HTMLResponse:
        """Generate error HTML response.

        Args:
            error_message: Main error message to display
            title: Page and header title
            error_detail: Optional detailed error information
            status_code: HTTP status code
            auto_close_seconds: Seconds before auto-closing window
            **kwargs: Additional template variables

        Returns:
            HTML response for failed authentication
        """
        error_detail_html = ""
        if error_detail:
            # Sanitize error detail to prevent XSS
            safe_detail = cls._sanitize_html(error_detail)
            error_detail_html = f'<div class="error-detail">{safe_detail}</div>'

        content = cls._ERROR_CONTENT.format(
            title=title,
            message=error_message,
            error_detail=error_detail_html,
            **kwargs,
        )

        script = cls._AUTO_CLOSE_SCRIPT.format(
            seconds=auto_close_seconds, milliseconds=auto_close_seconds * 1000
        )

        html = cls._BASE_TEMPLATE.format(
            title=title, header_color="#ef4444", content=content, script=script
        )

        return HTMLResponse(content=html, status_code=status_code)

    @classmethod
    def callback_error(
        cls,
        error: str | None = None,
        error_description: str | None = None,
        provider: OAuthProvider = OAuthProvider.GENERIC,
        **kwargs: Any,
    ) -> HTMLResponse:
        """Generate error response for OAuth callback errors.

        Args:
            error: OAuth error code
            error_description: OAuth error description
            provider: OAuth provider name
            **kwargs: Additional template variables

        Returns:
            HTML response for callback errors
        """
        if error == "access_denied":
            return cls.error(
                error_message=f"You denied access to {provider.value}.",
                title="Access Denied",
                error_detail=error_description,
                **kwargs,
            )
        elif error == "invalid_request":
            return cls.error(
                error_message="The authentication request was invalid.",
                title="Invalid Request",
                error_detail=error_description
                or "The OAuth request parameters were incorrect.",
                **kwargs,
            )
        elif error == "unauthorized_client":
            return cls.error(
                error_message="This application is not authorized.",
                title="Unauthorized Application",
                error_detail=error_description
                or "The client is not authorized to use this grant type.",
                **kwargs,
            )
        elif error == "unsupported_response_type":
            return cls.error(
                error_message="The authorization server does not support this response type.",
                title="Unsupported Response Type",
                error_detail=error_description,
                **kwargs,
            )
        elif error == "invalid_scope":
            return cls.error(
                error_message="The requested scope is invalid or unknown.",
                title="Invalid Scope",
                error_detail=error_description,
                **kwargs,
            )
        elif error == "server_error":
            return cls.error(
                error_message=f"The {provider.value} server encountered an error.",
                title="Server Error",
                error_detail=error_description or "Please try again later.",
                status_code=500,
                **kwargs,
            )
        elif error == "temporarily_unavailable":
            return cls.error(
                error_message=f"The {provider.value} service is temporarily unavailable.",
                title="Service Unavailable",
                error_detail=error_description or "Please try again later.",
                status_code=503,
                **kwargs,
            )
        else:
            # Generic error
            return cls.error(
                error_message=error_description
                or error
                or "An unknown error occurred.",
                title="Authentication Error",
                error_detail=f"Error code: {error}" if error else None,
                **kwargs,
            )

    @classmethod
    def _sanitize_html(cls, text: str) -> str:
        """Sanitize text for safe HTML display.

        Args:
            text: Text to sanitize

        Returns:
            Sanitized text safe for HTML display
        """
        # Basic HTML entity escaping
        replacements = {
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#x27;",
            "/": "&#x2F;",
        }

        for char, entity in replacements.items():
            text = text.replace(char, entity)

        return text
