import os
import sys

# This test exercises the `mega-bulk-sync-lowering` pass, which consumes the
# `mega.bulk_sync` op. We therefore load *both* the pass plugin (to register the
# pass) and the `mega` dialect plugin (so the op can be parsed).
# `TRITON_PLUGIN_PATHS` is a colon-separated list of shared libraries.
config.environment["TRITON_PLUGIN_PATHS"] = ":".join([
    os.path.join(config.triton_ext_binary_dir, "lib", "libmega.so"),
    os.path.join(config.triton_ext_binary_dir, "lib", "libmega_bulk_sync.so"),
])
print(
    f"ENV: "
    f"LD_LIBRARY_PATH={config.environment['LD_LIBRARY_PATH']} "
    f"TRITON_PLUGIN_PATHS={config.environment['TRITON_PLUGIN_PATHS']}",
    file=sys.stderr)
