triton_extension(tlMLanguage)

# The `tlM` language extension is a pure-Python DSL package (the `tlM` package
# under `python/`); there is nothing to compile. It is the Python frontend for
# the `mega` dialect: its builtins emit `mega` dialect ops through custom
# TritonOpBuilder methods registered by the `mega` dialect plugin (libmega.so).
# At runtime, put this `python/` directory on PYTHONPATH and load libmega.so via
# TRITON_PLUGIN_PATHS; importing `tlM` self-registers it as
# `triton.language.extra.tlM`.
#
# We register a marker target so `triton-ext-all-extensions` can depend on it.
add_custom_target(${PROJECT_NAME} ALL
    COMMENT "tlM language extension (pure-Python frontend for the mega dialect)")
