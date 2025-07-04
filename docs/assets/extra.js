// Custom JavaScript for Claude Code Proxy documentation

document.addEventListener('DOMContentLoaded', function() {
    // Add copy button functionality for code blocks
    document.querySelectorAll('pre code').forEach(function(block) {
        var button = document.createElement('button');
        button.className = 'copy-button';
        button.textContent = 'Copy';
        button.onclick = function() {
            navigator.clipboard.writeText(block.textContent).then(function() {
                button.textContent = 'Copied!';
                setTimeout(function() {
                    button.textContent = 'Copy';
                }, 2000);
            });
        };
        block.parentNode.appendChild(button);
    });

    // Add endpoint method styling
    document.querySelectorAll('code').forEach(function(code) {
        var text = code.textContent;
        if (/^(GET|POST|PUT|DELETE|PATCH)\s/.test(text)) {
            code.classList.add('api-method');
        }
    });

    // Add status code styling
    document.querySelectorAll('code').forEach(function(code) {
        var text = code.textContent;
        if (/^\d{3}$/.test(text)) {
            code.classList.add('status-code');
            var statusCode = parseInt(text);
            if (statusCode >= 200 && statusCode < 300) {
                code.classList.add('success');
            } else if (statusCode >= 400 && statusCode < 500) {
                code.classList.add('warning');
            } else if (statusCode >= 500) {
                code.classList.add('error');
            }
        }
    });

    // Auto-expand navigation for current page
    var currentUrl = window.location.pathname;
    document.querySelectorAll('.md-nav__link').forEach(function(link) {
        if (link.href && link.href.includes(currentUrl)) {
            var nav = link.closest('.md-nav__item');
            if (nav) {
                nav.classList.add('md-nav__item--active');
            }
        }
    });
});