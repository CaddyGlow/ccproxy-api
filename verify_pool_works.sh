#!/usr/bin/env
# Script to verify pool is working by comparing with and without pooling

echo "Pool Verification Script"
echo "======================="
echo ""
echo "This script will:"
echo "1. Start server with pooling enabled (default)"
echo "2. Make a request and check logs for 'Reusing existing connection'"
echo "3. Start server with pooling disabled"
echo "4. Make a request and verify no reuse messages"
echo ""
echo "Press Ctrl+C at any time to stop"
echo ""

# Function to make a test request
make_request() {
  curl -s -X POST http://localhost:8000/v1/messages \
    -H "Content-Type: application/json" \
    -d '{
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Say hello"}],
            "max_tokens": 10
        }' >/dev/null 2>&1
}

# Test 1: With pooling enabled (default)
echo "Starting server WITH pooling enabled..."
echo "--------------------------------------"
timeout 30s uv run ccproxy api 2>&1 | tee pool_enabled.log &
SERVER_PID=$!

# Wait for server to start
sleep 5

echo ""
echo "Making test requests..."
make_request
sleep 1
make_request
sleep 1
make_request

# Kill server
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null

# Check logs
echo ""
echo "Checking logs for pool reuse..."
if grep -q "Reusing existing connection" pool_enabled.log; then
  echo "✓ SUCCESS: Found connection reuse messages!"
  grep "Reusing existing connection" pool_enabled.log | head -3
else
  echo "✗ WARNING: No connection reuse found"
fi

echo ""
echo "Pool statistics:"
grep -E "(connections_created|connections_reused)" pool_enabled.log | tail -5

# Test 2: With pooling disabled
echo ""
echo ""
echo "Starting server WITHOUT pooling..."
echo "----------------------------------"
POOL_SETTINGS__ENABLED=false timeout 30s uv run python main.py 2>&1 | tee pool_disabled.log &
SERVER_PID=$!

# Wait for server to start
sleep 5

echo ""
echo "Making test requests..."
make_request
sleep 1
make_request
sleep 1
make_request

# Kill server
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null

# Check logs
echo ""
echo "Checking logs for pool disabled..."
if grep -q "Connection pooling is disabled" pool_disabled.log; then
  echo "✓ SUCCESS: Pool is disabled as expected"
fi

if grep -q "Reusing existing connection" pool_disabled.log; then
  echo "✗ ERROR: Found connection reuse with pool disabled!"
else
  echo "✓ SUCCESS: No connection reuse (as expected)"
fi

# Cleanup
rm -f pool_enabled.log pool_disabled.log

echo ""
echo "Verification complete!"
echo ""
echo "To run your own tests:"
echo "  With pool:    uv run python main.py"
echo "  Without pool: POOL_SETTINGS__ENABLED=false uv run python main.py"
