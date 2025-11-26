#!/bin/bash
# Export recommended IPEX/XPU environment variables for better performance

echo "Setting Intel XPU performance environment variables..."

# IPEX optimizations
export IPEX_XPU_ONEDNN_LAYOUT_OPT=1

# Disable verbose logging that might slow things down
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1

# Use device-side enqueue for better async performance  
export SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=1

# Disable kernel caching issues
export SYCL_CACHE_PERSISTENT=1

# Set tile configuration for Arc iGPU (8 subslices)
export ZE_AFFINITY_MASK=0

# Disable debug mode if accidentally enabled
unset ZE_ENABLE_DEBUG
unset SYCL_PI_TRACE

echo "Environment variables set:"
echo "  IPEX_XPU_ONEDNN_LAYOUT_OPT=$IPEX_XPU_ONEDNN_LAYOUT_OPT"
echo "  SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=$SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS"
echo "  SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=$SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS"
echo "  SYCL_CACHE_PERSISTENT=$SYCL_CACHE_PERSISTENT"
echo "  ZE_AFFINITY_MASK=$ZE_AFFINITY_MASK"
echo ""
echo "Now run your benchmark with these settings active:"
echo "  source set_xpu_env.sh"
echo "  .venv/bin/python benchmark_script.py --model Padim --category bottle --device xpu"