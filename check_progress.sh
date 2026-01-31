#!/bin/bash
# Check overnight progress

SSH_CMD="ssh -i ~/.ssh/imagegram_instance -p 5216 root@160.250.70.30"

echo "=== OVERNIGHT PROGRESS CHECK ==="
echo "Time: $(date)"
echo ""

echo "=== Process Status ==="
$SSH_CMD 'ps aux | grep overnight_run | grep -v grep | head -1'
echo ""

echo "=== GPU Status ==="
$SSH_CMD 'nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv'
echo ""

echo "=== Recent Log Output ==="
$SSH_CMD 'tail -30 /root/overnight_log.txt'
echo ""

echo "=== Output Files ==="
$SSH_CMD 'ls -lh /root/imagegram_overnight/ 2>/dev/null || echo "No output yet"'
