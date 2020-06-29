# -*- coding:utf-8 -*-
"""Load default configs when config module init."""
import os
from vega.core.common.user_config import DefaultConfig

DefaultConfig().load(os.path.dirname(os.path.abspath(__file__)))
