from pyiwfm.core.util.Utilities.version import Version
from pyiwfm.core.util.Utilities.util_revision import REVISION

VERSION = "1.0.0000"

class UtilVersion(Version):
    def __init__(self, version):
        super().__init__(version)

    @classmethod
    def get_util_version(cls):
        version = cls.version_new_from_components(VERSION, REVISION)

        return version.get_version()



        