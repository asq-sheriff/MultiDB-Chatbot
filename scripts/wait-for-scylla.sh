#!/bin/bash
# This script waits for the ScyllaDB cluster to be fully operational.

set -e

echo "⏳ Waiting for ScyllaDB cluster to become healthy..."
MAX_ATTEMPTS=30
ATTEMPT=1

while [ $ATTEMPT -le $MAX_ATTEMPTS ]; do
    # Use docker exec to run nodetool status inside the container
    # The grep command checks for "UN" which means "Up / Normal"
    NODES_UP=$(docker exec scylla-node1 nodetool status 2>/dev/null | grep -c "UN" || true)

    if [ "$NODES_UP" -eq "3" ]; then
        echo "✅ ScyllaDB cluster is fully up with $NODES_UP nodes!"
        exit 0
    fi

    echo "   Attempt $ATTEMPT/$MAX_ATTEMPTS: $NODES_UP/3 nodes are up. Retrying in 5 seconds..."
    sleep 5
    ATTEMPT=$((ATTEMPT + 1))
done

echo "❌ ScyllaDB cluster did not become healthy within the time limit."
exit 1