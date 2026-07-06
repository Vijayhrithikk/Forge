"""
Forge Acquisition Domain — safe, deterministic model asset acquisition.

Responsible for resolving model requirements from the registry,
downloading assets with resume support, verifying integrity via
checksums, and managing the filesystem cache.

Never loads models directly. Loading belongs to the Loader module.
"""
