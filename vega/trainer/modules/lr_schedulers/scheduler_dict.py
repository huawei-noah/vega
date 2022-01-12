#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Manage LrScheduler class."""

from vega.common import ClassFactory, ClassType


@ClassFactory.register(ClassType.LR_SCHEDULER)
class SchedulerDict(object):
    """Register and call VAEGANoptimizer class."""

    def __init__(self, optimizer, cfg):
        """Initialize."""
        self.cfg = cfg
        self.optimizer = optimizer
        for item in cfg['modules']:
            sub_optimizer_name = cfg[item]['optimizer']
            tem_lr_scheduler = ClassFactory.get_cls(
                ClassType.LR_SCHEDULER, cfg[item].type)
            sub_optimizer = getattr(optimizer, sub_optimizer_name)
            params = cfg[item].get("params", {})
            setattr(self, item, tem_lr_scheduler(sub_optimizer, **params))
