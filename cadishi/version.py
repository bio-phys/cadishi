# We handle version information in a single file as is commonly done, see e.g.
# https://packaging.python.org/single_source_version/#single-sourcing-the-version

# The 'ver' tuple is read by the setup.py, conf.py of the Sphinx documentation
# system and __init__.py of the package.
ver = (1,0,0)

def get_version_string():
    """Return the full version number."""
    return '.'.join(map(str,ver))

def get_short_version_string():
    """Return the version number without the patchlevel."""
    return '.'.join(map(str,ver[:-1]))
