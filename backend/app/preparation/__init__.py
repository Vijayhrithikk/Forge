"""
Forge Preparation Runtime — device, precision, memory, optimization management.

Transforms a loaded model into a training-ready PreparedModel.
The PreparedModel is immutable — Mission C (Training) consumes it
but never modifies it directly.
"""
