#!/bin/bash

echo "Running elasticseach instance's healthcheck..."
curl http://elasticsearch:9200/_cat/health

# Start the application server
echo "Starting Flask server..."
python server.py